from skimage.measure import block_reduce
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
import torchvision

import cv2
import json
import os
import sys
from os import listdir
from os.path import isfile, join, basename

sys.path.append("..")
import util

np.set_printoptions(threshold=sys.maxsize)

DIRPATH = os.getcwd()
DATAPATH = DIRPATH + '/data/'

BATCH_SIZE = 256
VALIDATION_SIZE = 1000

class RealImagesDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, transform=None):
        self.filenames = [join(images_dir, f) for f in listdir(images_dir) \
                          if isfile(join(images_dir, f)) and f.endswith(".JPG")]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = preprocess_camera_image(fname)
#         image = torch.tensor(image)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(basename(fname)[5])
        return image, label

class PartitionedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __add__(self, other):
        return self.dataset.append(other)

mnist_transforms = [
    torchvision.transforms.RandomAffine((-5,5), translate=(.1,.1), scale=(.5,1.)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: preprocess_data(x, noisy=False))
]

real_transforms = [
    torchvision.transforms.ToTensor()
]

def preprocess_data(x, noisy=False, blackwhite_factor=.15, thresh_factor=.3):
    # Threshold
    minn = np.min(x.detach().numpy())
    maxx = np.max(x.detach().numpy())
    rangee = maxx - minn
    black_thresh = minn + blackwhite_factor * rangee # every below this is "black"
    white_thresh = maxx - blackwhite_factor * rangee # every below this is "white"
    threshold_factor = np.random.rand() if noisy else thresh_factor
    threshold = threshold_factor * (white_thresh-black_thresh) + black_thresh
    return (x > threshold).float()

def preprocess_camera_image(jpg_path, black_threshold=.20, crop_margin=0.85):
    # Open with PIL
    with Image.open(jpg_path) as img_PIL:
        # Crop (to square, remove timestamp)
        width, height = img_PIL.size 
        target_size = crop_margin * height
        leftright_margin = (width - target_size) / 2
        topbottom_margin = (height - target_size) / 2
        area = (leftright_margin, topbottom_margin, width-leftright_margin, \
                       height-topbottom_margin) # rescale to square and remove time stamp
        img_PIL = img_PIL.crop(area)
        # Convert to Numpy
        img = np.asarray(img_PIL)
    # Grayscale (by averaging channels)
    img_gray = np.mean(img, axis=2)
    # Threshold
    minn = np.min(img_gray)
    maxx = np.max(img_gray)
    rng = maxx-minn
    thresh = black_threshold * rng + minn
    img_thresh = img_gray < thresh
    # Downsample (using max)
    img_small = block_reduce(img_thresh, block_size=(28, 28), func=np.max)[:28,:28]
    return img_small

def get_train_dataset():
    train_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=True, 
        transform=torchvision.transforms.Compose(mnist_transforms),
        download=True)
        
    return train_dataset

def get_test_dataset():
    test_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=False, 
        transform=torchvision.transforms.Compose(mnist_transforms),
        download=True)
    
    return test_dataset

def get_custom_dataset():
    with open('config.json') as config_file:
        config = json.load(config_file)

        NUMPY_SAVE_DIR = config['dataset']['NUMPY_SAVE_DIR']

    custom_dataset = []

    pwd_path = os.path.abspath(os.path.dirname(__file__))
    jpg_dir_path = os.path.join(pwd_path, NUMPY_SAVE_DIR)
    all_files = os.listdir(jpg_dir_path)
    npy_files = [files for files in all_files if ('npy' == files[-3:])]
    
    for numpy_file in npy_files:
        first_underscore = numpy_file.index("_")
        second_underscore = numpy_file.index("_", first_underscore+1)
        label = int(numpy_file[first_underscore+1:second_underscore])
        label_np = np.array([label])

        pwd_path = os.path.abspath(os.path.dirname(__file__))
        numpy_path = os.path.join(pwd_path, NUMPY_SAVE_DIR, numpy_file)

        img_np = np.load(numpy_path)
        img_tensor = torch.from_numpy(img_np).float()
        label_tensor = torch.from_numpy(label_np)

        custom_dataset.append([img_tensor, label_tensor])

    train_set = PartitionedDataset()
    for item in custom_dataset:
        train_set.__add__(item)

    return train_set

def fill_set_straight(source_dataset, target_set, target_set_size, index):
    for j in range(target_set_size):
        item = source_dataset.__getitem__(index)
        index = index + 1
        target_set.__add__(item)

    return index

def get_train_dataloader():
    train_dataset = get_train_dataset()

    train_set = PartitionedDataset()
    validation_set = PartitionedDataset()

    train_set_size = len(train_dataset) - VALIDATION_SIZE

    index = 0
    index = fill_set_straight(train_dataset, validation_set, VALIDATION_SIZE, index)
    index = fill_set_straight(train_dataset, train_set, train_set_size, index)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

    return train_loader, validation_loader

def get_custom_loader():
    custom_dataset = get_custom_dataset()
    total_size = len(custom_dataset)

    custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=total_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=total_size, shuffle=False)

    return custom_loader, validation_loader

def get_real_image_train_loader():
    real_images_loader = torch.utils.data.DataLoader(
        RealImagesDataset("./data/Custom/JPG",
                            transform=torchvision.transforms.Compose(real_transforms)),
                            batch_size=30, shuffle=True)
    
    validation_loader = torch.utils.data.DataLoader(
        RealImagesDataset("./data/Custom/JPG",
                            transform=torchvision.transforms.Compose(real_transforms)),
                            batch_size=30, shuffle=True)

    return real_images_loader, validation_loader

def get_real_image_test_loader():
    real_images_loader = torch.utils.data.DataLoader(
        RealImagesDataset("./data/Custom/JPG",
                            transform=torchvision.transforms.Compose(real_transforms)),
                            batch_size=30, shuffle=True)

    return real_images_loader

def get_test_dataloader():
    test_dataset = get_test_dataset()
    total_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=total_size, shuffle=False)
    
    return test_loader


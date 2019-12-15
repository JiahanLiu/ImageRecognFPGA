import numpy as np
from PIL import Image
from skimage.measure import block_reduce
import torch
import torchvision

from os import listdir
from os.path import isfile, join, basename

batch_size_train = 64
batch_size_test = 1000

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

def get_real_image_loader():
    real_images_loader = torch.utils.data.DataLoader(
        RealImagesDataset("./data/real_images/",
                            transform=torchvision.transforms.Compose(real_transforms)),
                            batch_size=30, shuffle=True)

    return real_images_loader

def get_real_image_train_loader():
    real_images_loader = torch.utils.data.DataLoader(
        RealImagesDataset("./data/first_split/",
                            transform=torchvision.transforms.Compose(real_transforms)),
                            batch_size=30, shuffle=True)

    return real_images_loader

def get_real_image_test_loader():
    real_images_loader = torch.utils.data.DataLoader(
        RealImagesDataset("./data/second_split/",
                            transform=torchvision.transforms.Compose(real_transforms)),
                            batch_size=30, shuffle=True)

    return real_images_loader

def get_train_loader():
    train_loader = torch.utils.data.DataLoader(
                        torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose(mnist_transforms)),
                        batch_size=batch_size_train, shuffle=True)

    return train_loader

def get_test_loader():
    test_loader = torch.utils.data.DataLoader(
                        torchvision.datasets.MNIST('./data', train=False, download=True,
                            transform=torchvision.transforms.Compose(mnist_transforms)),
                        batch_size=batch_size_test, shuffle=True)

    return test_loader
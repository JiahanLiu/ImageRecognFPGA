import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision

import cv2
import json
import os
import sys

sys.path.append("..")
import util

np.set_printoptions(threshold=sys.maxsize)

DIRPATH = os.getcwd()
DATAPATH = DIRPATH + '/data/'

BATCH_SIZE = 256
VALIDATION_SIZE = 1000

class PartitionedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __add__(self, other):
        return self.dataset.append(other)

def get_train_dataset():
    transforms = [
        torchvision.transforms.RandomAffine((-15,15), translate=(.2,.2)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: preprocess_data(x))
    ]

    train_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=True, 
        transform=torchvision.transforms.Compose(transforms),
        download=True)
        
    return train_dataset

def get_test_dataset():
    transforms = [
        torchvision.transforms.RandomAffine((-15,15), translate=(.2,.2)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: preprocess_data(x))
    ]

    test_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=False, 
        transform=torchvision.transforms.Compose(transforms),
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
        img_tensor = torch.from_numpy(img_np)
        label_tensor = torch.from_numpy(label_np)

        custom_dataset.append([img_tensor, label_tensor])

    return custom_dataset

def fill_set_straight(source_dataset, target_set, target_set_size, index):
    for j in range(target_set_size):
        item = source_dataset.__getitem__(index)
        index = index + 1
        target_set.__add__(item)

    return index

def preprocess_data(item):
    input_np = item.numpy()
    input_np = input_np * 255
    input_np = input_np.astype(np.uint8)
    input_np = input_np.reshape(28,28)
    (ret, input_np) = cv2.threshold(input_np, 0, 255, cv2.THRESH_OTSU)
    input_np = input_np.astype(np.float32)
    input_np = input_np.reshape(1,28,28)
    input_tensor = torch.from_numpy(input_np)
    
    return input_tensor

def process_and_fill_set_straight(source_dataset, target_set, target_set_size, index):
    for j in range(target_set_size):
        item = source_dataset.__getitem__(index)
        index = index + 1
        input_np = item[0].numpy()
        input_np = input_np[0] * 255
        input_np = input_np.astype(np.uint8)
        input_np = input_np.reshape(28,28)
        (ret, input_np) = cv2.threshold(input_np, 0, 255, cv2.THRESH_OTSU)
        input_np = input_np.astype(np.float32)
        input_np = input_np.reshape(1,28,28)
        input_tensor = torch.from_numpy(input_np)
        target_set.__add__((input_tensor, item[1]))
    
    # util.imshow(item_tensor, item[1])

    # print(type(item))
    # print(type((input_tensor, item[1])))

    # print(item_np)

    # img_np = item[0] * 255
    # img_np = img_np.astype(np.uint8)
    # print(img_np.dtype)
    # (ret, img_np) = cv2.threshold(img_np, 0, 255, cv2.THRESH_OTSU)
    # print(img_np)

    # util.imshow(img_np, item[1])

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

def get_train_dataloader_processed():
    train_dataset = get_train_dataset()

    train_set = PartitionedDataset()
    validation_set = PartitionedDataset()

    train_set_size = len(train_dataset) - VALIDATION_SIZE

    index = 0
    # index = process_and_fill_set_straight(train_dataset, validation_set, 1, index)
    index = fill_set_straight(train_dataset, validation_set, VALIDATION_SIZE, index)
    index = fill_set_straight(train_dataset, train_set, train_set_size, index)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=VALIDATION_SIZE, shuffle=False)

    return train_loader, validation_loader

def get_custom_loader():
    custom_dataset = get_custom_dataset()
    total_size = len(custom_dataset)

    custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=total_size, shuffle=False)

    return custom_loader

def get_test_dataloader():
    test_dataset = get_test_dataset()
    total_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=total_size, shuffle=False)
    
    return test_loader

def get_test_dataloader_processed():
    test_dataset = get_test_dataset()
    total_size = len(test_dataset)

    target_set = PartitionedDataset()
    
    index = 0
    index = process_and_fill_set_straight(test_dataset, target_set, total_size, index)

    test_loader = torch.utils.data.DataLoader(dataset=target_set, batch_size=total_size, shuffle=False)
    
    return test_loader


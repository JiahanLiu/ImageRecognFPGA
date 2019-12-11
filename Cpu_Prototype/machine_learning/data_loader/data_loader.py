import numpy as np
import torch
from torchvision import datasets, transforms

import json
import os
import sys

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
    train_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=True, 
        transform=transforms.ToTensor(),
        download=True)
        
    return train_dataset

def get_test_dataset():
    test_dataset = datasets.MNIST(
        root=DATAPATH, 
        train=False, 
        transform=transforms.ToTensor(),
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

    return custom_loader

def get_test_dataloader():
    test_dataset = get_test_dataset()
    total_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=total_size, shuffle=False)
    
    return test_loader


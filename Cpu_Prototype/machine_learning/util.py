import torch

import numpy as np
import matplotlib.pyplot as plt

import os

def save_architecture_to_file_closure(file_path):
    def save_architecture_to_file(architecture):
        if os.path.isfile(file_path):
            os.remove(file_path)
        torch.save(architecture.state_dict(), file_path)
    
    return save_architecture_to_file

def load_architecture_from_file_closure(file_path):
    def load_architecture_from_file(architecture, device):
        checkpoint = torch.load(file_path, map_location=device)
        architecture.load_state_dict(checkpoint)

    return load_architecture_from_file

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

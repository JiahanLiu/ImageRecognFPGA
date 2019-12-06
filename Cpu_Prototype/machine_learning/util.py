import torch

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
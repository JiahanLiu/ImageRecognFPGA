from data_loader import data_loader
from model import nn_architectures
import train
import util

import csv   
import getopt
import json
import random
import os
import sys

def train_single(width, depth): 
    with open('config.json') as config_file:
        config = json.load(config_file)
        EPOCH_SATURATION = config['machine_learning']['EPOCH_SATURATION']
        LEARNING_RATE = config['machine_learning']['LEARNING_RATE']
        TRANSFER_LEARNING_RATE = config['machine_learning']['LEARNING_RATE']
        MAX_EPOCHS = config['machine_learning']['MAX_EPOCHS']
        N_EPOCHS = config['machine_learning']['N_EPOCHS']

        SAVE_MODEL_DIR = config['results']['SAVE_MODEL_DIR']
        SAVE_MODEL_FILE_FC2 = config['results']['SAVE_MODEL_FILE_FC2']
        SAVE_MODEL_FILE_FC2 = SAVE_MODEL_FILE_FC2 + "_w_ " + str(width) + "_d_" + str(width) + ".sm"

    stop_at_epoch_saturation = train.stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)
    stop_at_N_epochs = train.stop_at_N_epochs_closure(N_EPOCHS)
    
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(pwd_path, SAVE_MODEL_DIR, SAVE_MODEL_FILE_FC2)
    save_architecture_to_file = util.save_architecture_to_file_closure(save_path)
    load_architecture_from_file = util.load_architecture_from_file_closure(save_path)

    train.train(nn_architectures.NetFC_X, data_loader.get_train_dataloader, data_loader.get_test_dataloader, stop_at_epoch_saturation, LEARNING_RATE, save_architecture_to_file, None, width, depth)
    acc = train.train(nn_architectures.NetFC_X, data_loader.get_real_image_train_loader, data_loader.get_real_image_test_loader, stop_at_N_epochs, TRANSFER_LEARNING_RATE, save_architecture_to_file, load_architecture_from_file, width, depth)

    return acc

def main():
    width = 100
    depth = 1

    options, remainder = getopt.getopt(sys.argv[1:], 'd:')
    for opt, arg in options:
        if opt in ('-d'):
            depth = int(arg)
    
    for width in range(30, 100, 10):
        acc = train_single(width, depth)
        row = [acc, width, depth]
        with open('hypersearch_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    train_single(width, depth)

if __name__ == "__main__":
    main()

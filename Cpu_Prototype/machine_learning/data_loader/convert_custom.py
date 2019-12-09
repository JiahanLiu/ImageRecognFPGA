import torch

import cv2
import numpy as np
from PIL import Image

import json
import os
import sys

sys.path.append("..")
import util

np.set_printoptions(threshold=sys.maxsize)

SCALE_FACTOR_REMOVE_TIME_STAMP = 0.9

def show_jpg(jpg_path, save_path):
    with Image.open(jpg_path) as img:
        target_size = 128, 128

        width, height = img.size 
        area = (0, 0, (width * 960/1280) * SCALE_FACTOR_REMOVE_TIME_STAMP, height * SCALE_FACTOR_REMOVE_TIME_STAMP) # rescale to square and remove time stamp
        img = img.crop(area)
        img.thumbnail(target_size, Image.ANTIALIAS) # make 128 by 128
        img = img.convert("L", dither = None) # convert to gray scale
        img_np = np.array(img)
        img_np = 255 - img_np # invert 
        # img_np = img_np -  img_np.min() # offset by subtracting min
        (ret, img_np) = cv2.threshold(img_np, 0, 255, cv2.THRESH_OTSU)
        img_np = img_np.reshape(1,128,128)
        np.save(save_path, img_np)
        # img_tensor = torch.from_numpy(img_np)
        # torch.save(img_tensor, save_path)
        # img_tensor = torch.load(save_path)
        # util.imshow(img_tensor)
        
def main(): 
    with open('../config.json') as config_file:
        config = json.load(config_file)

        JPG_DATA_DIR = config['dataset']['JPG_DATA_DIR']
        JPG_SAVE_DIR = config['dataset']['JPG_SAVE_DIR']

    jpg_name = "Test_7_0.JPG"
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    jpg_path = os.path.join(pwd_path, JPG_DATA_DIR, jpg_name)

    period_loc = jpg_name.index('.')
    numpy_file = jpg_name[:period_loc]
    print(numpy_file)
    save_path = os.path.join(pwd_path, JPG_SAVE_DIR, numpy_file)
    # print(pwd_path)
    # print(jpg_path)
    # print(save_path)
    show_jpg(jpg_path, save_path)

if __name__ == "__main__":
    main()

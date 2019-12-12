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

def jpg_to_numpy(jpg_path, save_path, np_print=False):
    with Image.open(jpg_path) as img:
        target_height = 28
        target_width = 28
        target_size = target_height, target_width

        width, height = img.size 
        area = (0, 0, (width * 960/1280) * SCALE_FACTOR_REMOVE_TIME_STAMP, height * SCALE_FACTOR_REMOVE_TIME_STAMP) # rescale to square and remove time stamp
        img = img.crop(area)
        img.thumbnail(target_size, Image.ANTIALIAS) # make 28 by 28
        img = img.convert("L", dither = None) # convert to gray scale
        img_np = np.array(img)
        if(np_print):
            print(img_np)
        img_np = 255 - img_np # invert 
        # img_np = img_np -  img_np.min() # offset by subtracting min
        (ret, img_np) = cv2.threshold(img_np, 0, 255, cv2.THRESH_OTSU)
        img_np = img_np.astype(np.float32)
        img_np = img_np.reshape(1,target_height,target_width)
        np.save(save_path, img_np)

        img_tensor = torch.from_numpy(img_np)

        # torch.save(img_tensor, save_path)
        # img_tensor = torch.load(save_path)
        # util.imshow(img_tensor)
    
    return img_tensor
        
def main(): 
    with open('../config.json') as config_file:
        config = json.load(config_file)

        JPG_DATA_DIR = config['dataset']['JPG_DATA_DIR']
        NUMPY_SAVE_DIR = config['dataset']['NUMPY_SAVE_DIR']

    pwd_path = os.path.abspath(os.path.dirname(__file__))
    jpg_dir_path = os.path.join(pwd_path, JPG_DATA_DIR)
    all_files = os.listdir(jpg_dir_path)
    jpg_files = [files for files in all_files if ('JPG' == files[-3:])]

    img_tensors_list = []
    for jpg_name in jpg_files:
        jpg_path = os.path.join(pwd_path, JPG_DATA_DIR, jpg_name)
        period_loc = jpg_name.index('.')
        numpy_file = jpg_name[:period_loc]
        save_path = os.path.join(pwd_path, NUMPY_SAVE_DIR, numpy_file)
        img_tensor = jpg_to_numpy(jpg_path, save_path)
        # if("Test_9_0.JPG" == jpg_name):
        #     jpg_to_numpy(jpg_path, save_path, True)

        first_underscore = jpg_name.index("_")
        second_underscore = jpg_name.index("_", first_underscore+1)
        period = jpg_name.index(".")
        batch_num = int(jpg_name[second_underscore+1:period])
        target_batch_num = 19
        if(batch_num == target_batch_num):
            img_tensors_list.append(img_tensor)

    util.imshow_list(img_tensors_list)

if __name__ == "__main__":
    main()

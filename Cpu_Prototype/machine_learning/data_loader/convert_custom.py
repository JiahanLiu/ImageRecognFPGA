import torch

import cv2
import numpy as np
from PIL import Image
from skimage.measure import block_reduce

import json
import os
import sys

sys.path.append("..")
import util

np.set_printoptions(threshold=sys.maxsize)

SCALE_FACTOR_REMOVE_TIME_STAMP = 0.9

def preprocess_camera_image(jpg_path, save_path, black_threshold=.20, crop_margin=0.85):
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
    # Treshold
    minn = np.min(img_gray)
    maxx = np.max(img_gray)
    rng = maxx-minn
    thresh = black_threshold * rng + minn
    img_thresh = 1.0 * (img_gray < thresh)
    # Downsample (using max)
    img_small = block_reduce(img_thresh, block_size=(28, 28), func=np.max)[:28,:28]

    img_small.astype(np.single)
    # print(img_small.dtype)
    
    # print(img_small.shape)

    np.save(save_path, img_small)

    img_np_view = img_small.reshape(1,28,28)
    img_tensor = torch.from_numpy(img_np_view)

    return img_tensor

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
        img_np = img_np.reshape(target_height,target_width)
        np.save(save_path, img_np)

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
        img_tensor = preprocess_camera_image(jpg_path, save_path)
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

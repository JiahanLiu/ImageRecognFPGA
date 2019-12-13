import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import block_reduce


if len(sys.argv) <= 2:
    print("Usage: python display_preprocessed_iamge.py <path to data> <path to img>")
    exit(0)


filepath = sys.argv[1]
imgpath = sys.argv[2]




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
    # Treshold
    minn = np.min(img_gray)
    maxx = np.max(img_gray)
    rng = maxx-minn
    thresh = black_threshold * rng + minn
    img_thresh = img_gray < thresh
    # Downsample (using max)
    img_small = block_reduce(img_thresh, block_size=(28, 28), func=np.max)[:28,:28]
    return img_small




with open(filepath, "r") as f:
    contents = f.read()
    values = [float(s.strip()) for s in contents.split(",")]
    c_img = np.array(values).reshape(28,28)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(c_img, cmap='gray')
    plt.title("C Preprocessed Image")
    plt.subplot(1,2,2)
    py_img = preprocess_camera_image(imgpath)
    plt.imshow(py_img, cmap='gray')
    plt.title("Python Preprocessed Image")
    plt.show()

from data_loader import data_loader

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import block_reduce
import torch

import numpy as np
import time

model_path_mnist_only = "./model/model.pt"
model_path_real_data = "./model/FC2.sm"

def accuracy(architecture, data_loader):
    correct = 0
    architecture.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x
            test_y = test_y
            output = architecture(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(data_loader.dataset)

    return acc

def test(model, test_loader):
    model.eval()
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            image = images[i].view(1, 784)
            with torch.no_grad():
                log_probs = model(image)
            probs_tnsr = torch.exp(log_probs)
            probabilities = list(probs_tnsr.numpy()[0])
            pred_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    acc = 100 * correct_count / all_count # model accuracy
    return acc, correct_count, all_count

def try_model(model_path):
    model = torch.load(model_path)

    real_test_loader = data_loader.get_real_image_loader()
    
    acc = accuracy(model, real_test_loader)

    print("Accuracy: " + str(acc))

def preprocess_camera_image(jpg_path, black_threshold=.20, crop_margin=0.75):
    # Open with PIL
    with Image.open(jpg_path) as img_PIL:
        # Crop (to square, remove timestamp)
        width, height = img_PIL.size 
        # print(str(width) + " " + str(height))
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
    img_small = block_reduce(img_thresh, block_size=(11, 11), func=np.max)[:28,:28]
    # imgplot = plt.imshow(img_small)
    # plt.show()
    return img_small

def fun(model_path):
    model = torch.load(model_path)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()

    else:
        rval = False

    # while rval:
    #     cv2.imshow("preview", frame)
    #     rval, frame = vc.read()
    #     key = cv2.waitKey(20)
    #     if key == 27: # exit on ESC
    #         break
    
    img_path = "opencv_frame.jpg"
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        cv2.imwrite(img_path, frame)

        time.sleep(1)
        image = preprocess_camera_image(img_path)
        image_tnsr = torch.from_numpy(image).float().view(1,784)
        with torch.no_grad():
            log_probs = model(image_tnsr)
        probs_tnsr = torch.exp(log_probs)
        probabilities = list(probs_tnsr.numpy()[0])
        pred_label = probabilities.index(max(probabilities))

        print(pred_label)

    cv2.destroyWindow("preview")
    vc.release()
    
    print("Done")

def main():
    # fun(model_path_real_data)
    try_model(model_path_real_data)

if __name__ == "__main__":
    main()

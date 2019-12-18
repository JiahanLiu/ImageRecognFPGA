from data_loader import data_loader

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import block_reduce
import torch
import torch.nn as nn

import numpy as np
import pickle
import time

model_path_mnist_only = "./model/model.pt"
model_path_real_data = "./model/model_real_data.pt"

numpy_eval_save_file = "model/numpy_eval.pkl"

# # Hidden layers' weights/biases
hidden_layer_weights = []
hidden_layer_biases = []
num_hidden_layers = 0 # count of hidden layers
# Output layer's weight/bias
output_layer_weight = None
output_layer_bias = None

def evaluate_numpy_nn(x):
    global hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias

    # Hidden layers
    for h in range(num_hidden_layers):
        # print(hidden_layer_weights[h].shape)
        x = np.matmul(hidden_layer_weights[h], x) + hidden_layer_biases[h] # H: W*x + b
        x = np.maximum(x, 0) # ReLU
    # Output layer
    x = np.matmul(output_layer_weight, x) + output_layer_bias # Output: W*x + b

    return x

def load_numpy_model():
    global hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias
    with open(numpy_eval_save_file, 'rb') as f:
        hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias = pickle.load(f)

def parse_pytorch_model(mdl):
    hidden_layer_weights = []
    hidden_layer_biases = []
    num_hidden_layers = 0
    output_layer_weight = None
    output_layer_bias = None

    num_modules = len(mdl)
    i = 0
    while i < num_modules-1:
        module = mdl[i]; i += 1
        next_module = mdl[i]; i += 1

        # Expect a linear module first
        assert isinstance(module, nn.modules.linear.Linear), "Expected Linear module"

        # Make sure following module is ReLU or LogSoftmax
        assert isinstance(next_module, nn.modules.activation.LogSoftmax) or \
                isinstance(next_module, nn.modules.activation.ReLU), "Expected ReLU or LogSoftmax module"

        # Make sure LogSoftmax is only at the end
        if isinstance(next_module, nn.modules.activation.LogSoftmax):
            assert i >= num_modules-1

        # Make sure last model is LogSoftmax
        if i == num_modules-1:
            assert isinstance(next_module, nn.modules.activation.LogSoftmax), "LogSoftmax should be last module."

        # Parse Linear module
        weight = module.weight.detach().numpy()
        bias = module.bias.detach().numpy()
        if isinstance(next_module, nn.modules.activation.ReLU): # is hidden layer
            hidden_layer_weights.append(weight)
            hidden_layer_biases.append(bias)
            num_hidden_layers += 1
        else: # is output layer
            output_layer_weight = weight
            output_layer_bias = bias

    with open(numpy_eval_save_file, 'bw') as f:
        pickle.dump([hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias], f)

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
    acc = test(model, real_test_loader)

    print("Accuracy: " + str(acc))

def test_np(test_loader):
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            image = images[i].view(1, 784)
            image_np = image.numpy()
            image_np = image_np.reshape(784)
            numpy_output = evaluate_numpy_nn(image_np)
            pred_label = np.argmax(numpy_output)
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    acc = 100 * correct_count / all_count # model accuracy
    return acc, correct_count, all_count

def try_model_np():
    load_numpy_model()
    convert_weights_to_int()

    real_test_loader = data_loader.get_real_image_loader()
    acc = test_np(real_test_loader)

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

def camera(model_path):
    model = torch.load(model_path)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
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

def camera_numpy(model_path):
    load_numpy_model()
    convert_weights_to_int()

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
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
        image = image.reshape(784)
        numpy_output = evaluate_numpy_nn(image)
        numpy_output = numpy_output * 1000
        prediction = np.argmax(numpy_output)
        
        print(prediction)
        # print(numpy_output)

    cv2.destroyWindow("preview")
    vc.release()
    
    print("Done")

def convert_weights_to_int():
    global hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias
    multiplier = 1024

    hidden_layer_weights[0] = hidden_layer_weights[0] * multiplier
    hidden_layer_weights[0] = hidden_layer_weights[0].astype(np.int64)

    hidden_layer_weights[1] = hidden_layer_weights[1] * multiplier
    hidden_layer_weights[1] = hidden_layer_weights[1].astype(np.int64)

    hidden_layer_biases[0] = hidden_layer_biases[0] * multiplier
    hidden_layer_biases[0] = hidden_layer_biases[0].astype(np.int64)

    hidden_layer_biases[1] = hidden_layer_biases[1] * multiplier
    hidden_layer_biases[1] = hidden_layer_biases[1].astype(np.int64)

    output_layer_weight = output_layer_weight * multiplier
    output_layer_weight = output_layer_weight.astype(np.int64)

    output_layer_bias = output_layer_bias * multiplier
    output_layer_bias = output_layer_bias.astype(np.int64)

def print_comma_seperated_list(list):
    output = "{"
    for item in list[:-1]:
        output += str(item) + ", "
    output += str(list[-1])
    output += "},"

    print(output)

def get_w_1(row):
    global hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias

    print_comma_seperated_list(hidden_layer_weights[0][row])


def get_weights(row):
    global hidden_layer_weights, hidden_layer_biases, num_hidden_layers, output_layer_weight, output_layer_bias

    get_w_1(row)

    # print(num_hidden_layers) # 2
    # print(len(hidden_layer_weights)) # 2
    # print(hidden_layer_weights[0].shape) # 200 X 784
    # print(hidden_layer_weights[1].shape) # 200 X 200

    # print(len(hidden_layer_biases)) # 2
    # print(hidden_layer_biases[0].shape) #200
    # print(hidden_layer_biases[1].shape) #200

    # print(output_layer_weight.shape) # 10, 200
    # print(output_layer_bias.shape) # 10

def fpga_int64_weights(row):
    load_numpy_model()
    convert_weights_to_int()
    get_weights(row)


def main():
    # camera(model_path_real_data)
    # camera_numpy(model_path_real_data)

    # try_model(model_path_real_data)
    # try_model_np()

    # fpga_int32_weights(model_path)
    fpga_int64_weights(6)

if __name__ == "__main__":
    main()

import model
import util

import numpy as np
import matplotlib.pyplot as plt
import torch

def accuracy(architecture, data_loader, device):
    correct = 0
    architecture.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            output = architecture(test_x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(test_y.view_as(pred)).sum().item()
        acc = (100 * correct) / len(data_loader.dataset)

    return acc

def single_test(architecture, data_loader, device, target_index):
    saved_test_x = None

    current_index = 0
    architecture.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            current_index = current_index + 1
            x_1 = test_x[target_index]
            y_1 = test_y[target_index]
            
            x_1 = x_1.to(device)
            y_1 = y_1.to(device)
            output = architecture(x_1)
            pred = output.argmax(dim=1, keepdim=True)
            title = "Pred: " + str(pred) + " | Correct: " + str(y_1)
            util.imshow(x_1, title)

        # # print(type(data_loader))
        # for i in range(target_index+1):
        #     print("iterated")
        #     batch_x_y = next(iter(data_loader))
        # batch_x = batch_x_y[0]
        # batch_y = batch_x_y[1]
        # x_1 = batch_x[0]
        # y_1 = batch_y[0]
        
        # x_1 = x_1.to(device)
        # y_1 = y_1.to(device)
        # output = architecture(x_1)
        # pred = output.argmax(dim=1, keepdim=True)
        # title = "Pred: " + str(pred) + " | Correct: " + str(y_1)
        # util.imshow(x_1, title)

def loss(architecture, data_loader, loss_fn, device):
    architecture.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            outputs = architecture(test_x)
            loss = loss_fn(outputs, test_y)

    return loss 

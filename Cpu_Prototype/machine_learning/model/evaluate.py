import model

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

def loss(architecture, data_loader, loss_fn, device):
    architecture.eval()
    with torch.no_grad():
        for test_x, test_y in data_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            outputs = architecture(test_x)
            loss = loss_fn(outputs, test_y)

    return loss 
    
import torch as torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class NetFC_2(nn.Module):

    def __init__(self, width):
        super(NetFC_2, self).__init__()

        self.z1 = nn.Linear(784, 100)
        self.z2 = nn.Linear(100, 100)
        self.z3_output = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.z1(x))
        x = F.relu(self.z2(x))
        x = self.z3_output(x)
        x = F.log_softmax(x, dim=1)

        return x

class NetFC_5(nn.Module):

    def __init__(self):
        super(NetFC_5, self).__init__()

        self.z1 = nn.Linear(784, 100)
        self.z2 = nn.Linear(100, 100)
        self.z3 = nn.Linear(100, 50)
        self.z4 = nn.Linear(50, 50)
        self.z5_output = nn.Linear(50, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.z1(x))
        x = F.relu(self.z2(x))
        x = F.relu(self.z3(x))
        x = F.relu(self.z4(x))
        x = self.z5_output(x)
        x = F.log_softmax(x, dim=1)

        return x

class NetCNN_convrelu3_relu3(nn.Module):
    def __init__(self):
        super(NetCNN_convrelu3_relu3, self).__init__()

        self.conv_block = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) 
            )

        self.linear_block = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128*7*7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
            )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        x = F.log_softmax(x, dim=1)
        
        return x    

def parameter_init(model):
    if type(model) == nn.Linear:
        torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        model.bias.data.fill_(0.01)

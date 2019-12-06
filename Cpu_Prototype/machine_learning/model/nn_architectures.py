import torch as torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class NetFC_2(nn.Module):

    def __init__(self):
        super(NetFC_2, self).__init__()

        self.z1 = nn.Linear(784, 200)
        self.z2 = nn.Linear(200, 200)
        self.z3_output = nn.Linear(200, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.z1(x))
        x = F.relu(self.z2(x))
        x = self.z3_output(x)
        x = F.log_softmax(x, dim=1)

        return x

def parameter_init(model):
    if type(model) == nn.Linear:
        torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        model.bias.data.fill_(0.01)

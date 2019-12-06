from model import evaluate
from model import nn_architectures

import torch
import torch.nn as nn
import torch.optim as optim

import random

class Model:
    def __init__(self, network_architecture, train_loader, validation_loader, test_loader, learning_rate):
        self.DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            print("Federated Using Cuda")
        torch.manual_seed(random.random() * 100)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.architecture = network_architecture().to(device=self.DEVICE)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.architecture.parameters(), lr=learning_rate)

        nn_architectures.parameter_init(self.architecture)

    def train(self):
        self.architecture.train()
        for batch_idx, (train_x, train_y) in enumerate(self.train_loader):
            train_x = train_x.to(self.DEVICE)
            train_y = train_y.to(self.DEVICE)
            self.optimizer.zero_grad()
            outputs = self.architecture(train_x)
            loss = self.loss_fn(outputs, train_y)
            loss.backward()
            self.optimizer.step()

        return (loss, self.architecture.parameters())

    def get_test_accuracy(self):
        test_acc = evaluate.accuracy(self.architecture, self.test_loader, self.DEVICE)
        return test_acc

    def get_validation_accuracy(self):
        val_acc = evaluate.accuracy(self.architecture, self.validation_loader, self.DEVICE)
        return val_acc

    def get_loss(self):
        loss = evaluate.loss(self.architecture, self.train_loader, self.loss_fn, self.DEVICE)
        return loss

    def get_architecture(self):
        return self.architecture
        
    def set_device(self, device):
        if torch.cuda.is_available():
            self.DEVICE = torch.device(device)

    def get_device(self):
        return self.DEVICE

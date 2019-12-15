from model import evaluate
from model import nn_architectures

import torch
import torch.nn as nn
import torch.optim as optim

import random

class Model:
    def __init__(self, network_architecture, train_loader, validation_loader, test_loader, learning_rate, width, depth):
        self.DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            print("Federated Using Cuda")
        torch.manual_seed(random.random() * 100)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.learning_rate = learning_rate
        self.architecture = network_architecture(width, depth).to(device=self.DEVICE)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.architecture.parameters(), lr=learning_rate)

        nn_architectures.parameter_init(self.architecture)

    def train(self):
        for images, labels in self.train_loader:
            images.to(device=self.DEVICE)
            labels.to(device=self.DEVICE)
            images = images.view(images.shape[0], -1) # flatten image into 1D vector
            self.optimizer.zero_grad()
            outputs = self.architecture(images)
            loss = self.loss_fn(outputs, labels)
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

    def set_architecture(self, architecture):
        self.architecture = architecture.to(device=self.DEVICE)

        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.architecture.parameters(), lr=self.learning_rate)
        
    def set_device(self, device):
        if torch.cuda.is_available():
            self.DEVICE = torch.device(device)

    def get_device(self):
        return self.DEVICE

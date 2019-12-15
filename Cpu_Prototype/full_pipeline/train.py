from data_loader import data_loader
import evaluate
from model import model

import torch
import torch.nn as nn
import torch.optim as optim

import csv   
import getopt
import random
import sys

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Federated Using Cuda")
torch.manual_seed(random.random() * 100)

def test(model, test_loader, device):
    model.eval()
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        for i in range(len(labels)):
            image = images[i].view(1, 784)
            with torch.no_grad():
                log_probs = model(image)
            probs_tnsr = torch.exp(log_probs).cpu()
            probabilities = list(probs_tnsr.numpy()[0])
            pred_label = probabilities.index(max(probabilities))
            true_label = labels.cpu().numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    acc = 100 * correct_count / all_count # model accuracy
    return acc, correct_count, all_count

    return acc

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    cumulative_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.shape[0], -1) # flatten image into 1D vector
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        cumulative_loss += loss.item()
    return cumulative_loss / len(train_loader)

def train_model(model, optimizer, criterion, train_loader, test_loader, model_path, num_epochs, device):
    training_results = []
    testing_results = []
    for epoch in range(1, num_epochs+1):
        print("Epoch %d..." % epoch)
        avg_loss = train(model, optimizer, criterion, train_loader, device)
        percent_correct_train, num_correct_train, count_train = test(model, train_loader, device)
        percent_correct_test, num_correct_test, count_test = test(model, test_loader, device)
        training_results.append(percent_correct_train)
        testing_results.append(percent_correct_test)
        print("\tTraining Loss: %.4f" % avg_loss)
        print("\tTraining Error: %.2f%% (%d/%d)" % (percent_correct_train, num_correct_train, count_train))
        print("\tTesting Error : %.2f%% (%d/%d)" % (percent_correct_test, num_correct_test, count_test))

    torch.save(model, model_path)

def train_base(base_model_path, num_epochs, num_hidden_layers, hidden_layer_width):
    net = model.build_simple_neural_net(num_hidden_layers, hidden_layer_width, input_size=28*28, output_size=10, device=DEVICE)
    model.parameter_init(net)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    train_model(net, optimizer, criterion, train_loader, test_loader, base_model_path, num_epochs, DEVICE)

    return net

def train_transfer(base_model_path, new_model_path, num_epochs):
    net = torch.load(base_model_path)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    train_loader = data_loader.get_real_image_train_loader()
    test_loader = data_loader.get_real_image_test_loader()

    train_model(net, optimizer, criterion, train_loader, test_loader, new_model_path, num_epochs, DEVICE)

    return net

def try_architecture(num_hidden_layers, hidden_layer_width):
    base_model_path = "./data/model.pt"
    new_model_path = "./data/model_real_data.pt"

    train_base(base_model_path, 50, num_hidden_layers, hidden_layer_width)
    net = train_transfer(base_model_path, new_model_path, 10)
    acc = evaluate.try_model(net, DEVICE)

    return acc

def main():
    start_search = 0
    options, remainder = getopt.getopt(sys.argv[1:], 'l:')
    for opt, arg in options:
        if opt in ('-l'):
            start_search = int(arg)
    
    for num_hidden_layers in range(start_search, start_search+2, 1):
        for hidden_layer_width in range(30, 100, 5):
            acc = try_architecture(num_hidden_layers, hidden_layer_width)
            row = [acc, num_hidden_layers, hidden_layer_width]
            with open('results.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)

if __name__ == "__main__":
    main()
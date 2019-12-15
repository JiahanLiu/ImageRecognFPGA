import torch
import torch.nn as nn

def build_simple_neural_net(num_hidden_layers, hidden_layer_width, input_size, output_size, device):
    modules = []
    
    for hidden_layer in range(num_hidden_layers):
        # Hidden layer
        if hidden_layer == 0: # first layer
            modules.append(nn.Linear(input_size, hidden_layer_width))
        else:
            modules.append(nn.Linear(hidden_layer_width, hidden_layer_width))
        # ReLU
        modules.append(nn.ReLU())
        
    # Output layer
    modules.append(nn.Linear(hidden_layer_width, output_size))
    
    # Softmax
    modules.append(nn.LogSoftmax(dim=1))
    
    return nn.Sequential(*modules).to(device)

def parameter_init(model):
    if type(model) == nn.Linear:
        torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        model.bias.data.fill_(0.01)
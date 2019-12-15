from data_loader import data_loader
from model import model, nn_architectures, evaluate
import util

import json
import os.path

DEBUG = True

def print_results(epoch_n, loss, validation_accuracy, acc):
    if DEBUG:
        print("Epoch: " + epoch_n + " | Loss: " + loss + " | ValAcc: " + validation_accuracy + " | Acc: " + acc)

def train(network_architecture, get_train_loader, get_test_loader, end_function, learning_rate, save_model_function, load_model_function, width, depth):
    train_loader, validation_loader = get_train_loader()
    test_loader = get_test_loader()
    net = model.Model(network_architecture, train_loader, validation_loader, test_loader, learning_rate, width, depth)
    if(None != load_model_function):
        print("Doing transfer learning")
        net.set_architecture(load_model_function(net.get_device()))

    end_condition = False
    epoch_n = 0
    losses = []
    validation_accuracies = []
    accuracies = []
    while False == end_condition:
        (loss_i, local_param_i) = net.train()
        loss = net.get_loss()
        validation_accuracy = net.get_validation_accuracy()
        acc = net.get_test_accuracy()

        losses.append(loss)
        validation_accuracies.append(validation_accuracy)
        accuracies.append(acc)
        end_condition, optimal_epoch = end_function(epoch_n, validation_accuracies)

        if(optimal_epoch == epoch_n):
            if(DEBUG):
                print("Saving Model")
            save_model_function(net.get_architecture())
        print_results(epoch_n=str(epoch_n), loss=str(loss.item()), validation_accuracy=str(validation_accuracy), acc=str(acc))

        epoch_n = epoch_n + 1
    
    return acc

def test_saved_model(network_architecture, get_train_loader, get_test_loader, learning_rate, load_model_function):
    train_loader, validation_loader = get_train_loader()
    test_loader = get_test_loader()
    net = model.Model(network_architecture, train_loader, validation_loader, test_loader, learning_rate)
    
    load_model_function(net.get_architecture(), net.get_device())
    loss = net.get_loss()
    validation_accuracy = net.get_validation_accuracy()
    acc = net.get_test_accuracy()

    print_results(epoch_n="N/A", loss=str(loss.item()), validation_accuracy=str(validation_accuracy), acc=str(acc))

def test_single_saved_model(network_architecture, get_train_loader, get_test_loader, learning_rate, load_model_function, target_index):
    train_loader, validation_loader = get_train_loader()
    test_loader = get_test_loader()
    net = model.Model(network_architecture, train_loader, validation_loader, test_loader, learning_rate)
    
    load_model_function(net.get_architecture(), net.get_device())
    evaluate.single_test(net.get_architecture(), test_loader, net.get_device(), target_index)

def stop_at_N_epochs_closure(N_epoch):
    def end_N_epochs(n_epoch, measures):
        if(n_epoch < N_epoch - 1):
            return False, N_epoch-1
        else:
            return True, N_epoch-1

    return end_N_epochs

def stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION):
    def end_epoch_saturation(n_epoch, measures):
        highest_measure_index = 0
        highest_measure = measures[highest_measure_index]
        for i in range(len(measures)):
            if measures[i] > highest_measure:
                highest_measure_index = i
                highest_measure = measures[highest_measure_index]
        
        if len(measures) > MAX_EPOCHS:
            return True, highest_measure_index

        if len(measures) <= EPOCH_SATURATION:
            return False, highest_measure_index

        if(highest_measure_index + EPOCH_SATURATION > len(measures)):
            return False, highest_measure_index
        else:
            return True, highest_measure_index
    
    return end_epoch_saturation

def main(): 
    with open('config.json') as config_file:
        config = json.load(config_file)
        EPOCH_SATURATION = config['machine_learning']['EPOCH_SATURATION']
        LEARNING_RATE = config['machine_learning']['LEARNING_RATE']
        TRANSFER_LEARNING_RATE = config['machine_learning']['LEARNING_RATE']
        MAX_EPOCHS = config['machine_learning']['MAX_EPOCHS']
        N_EPOCHS = config['machine_learning']['N_EPOCHS']

        SAVE_MODEL_DIR = config['results']['SAVE_MODEL_DIR']
        SAVE_MODEL_FILE_FC2 = config['results']['SAVE_MODEL_FILE_FC2']

    stop_at_epoch_saturation = stop_at_epoch_saturation_closure(MAX_EPOCHS, EPOCH_SATURATION)
    stop_at_N_epochs = stop_at_N_epochs_closure(N_EPOCHS)
    
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(pwd_path, SAVE_MODEL_DIR, SAVE_MODEL_FILE_FC2)
    save_architecture_to_file = util.save_architecture_to_file_closure(save_path)
    load_architecture_from_file = util.load_architecture_from_file_closure(save_path)

    width = 100
    depth = 2
    train(nn_architectures.NetFC_X, data_loader.get_train_dataloader, data_loader.get_test_dataloader, stop_at_epoch_saturation, LEARNING_RATE, save_architecture_to_file, None, width, depth)
    acc = train(nn_architectures.NetFC_X, data_loader.get_real_image_train_loader, data_loader.get_real_image_test_loader, stop_at_N_epochs, TRANSFER_LEARNING_RATE, save_architecture_to_file, load_architecture_from_file, width, depth)
    print("Final Accuracy" + str(acc))


    # test_saved_model(nn_architectures.NetFC_X, data_loader.get_train_dataloader, data_loader.get_test_dataloader, LEARNING_RATE, load_architecture_from_file)
    # test_saved_model(nn_architectures.NetFC_X, data_loader.get_train_dataloader, data_loader.get_custom_loader, LEARNING_RATE, load_architecture_from_file)

    temp_index = 1
    # test_single_saved_model(nn_architectures.NetFC_2, data_loader.get_train_dataloader, data_loader.get_test_dataloader, LEARNING_RATE, load_architecture_from_file, target_index=temp_index)
    # test_single_saved_model(nn_architectures.NetFC_2, data_loader.get_train_dataloader, data_loader.get_custom_loader, LEARNING_RATE, load_architecture_from_file, target_index=temp_index)
    
    # data_loader.get_custom_dataset()
    # data_loader.get_custom_loader()

    # data_loader.get_train_dataloader_processed()

    # data_loader.get_train_dataloader()
    # data_loader.get_custom_loader()


if __name__ == "__main__":
    main()

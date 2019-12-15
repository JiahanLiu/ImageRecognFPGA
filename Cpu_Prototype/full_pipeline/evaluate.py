from data_loader import data_loader

import torch

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        acc = (100 * correct) / len(test_loader.dataset)
        #     for i in range(len(labels)):
        #         image = images[i].view(1, 784)
        #         log_probs = model(image)
        #         pred_label = log_probs.argmax(dim=1, keepdim=True)
        #         true_label = labels.cpu().numpy()[i]
        #         if(true_label == pred_label):
        #             correct_count += 1
        #         all_count += 1
        # acc = 100 * correct_count / all_count # model accuracy
        return acc

def try_model(net, device):
    real_test_loader = data_loader.get_real_image_loader()
    
    acc, correct_count, all_count = test(net, real_test_loader, device)

    print("Accuracy: " + str(acc))

    return acc

def try_model_path(model_path, device):
    model = torch.load(model_path)

    real_test_loader = data_loader.get_real_image_loader()
    
    acc, correct_count, all_count = test(model, real_test_loader, device)

    print("Accuracy: " + str(acc))

    return acc
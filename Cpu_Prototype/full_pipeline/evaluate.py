from data_loader import data_loader

import torch

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
            probs_tnsr = torch.exp(log_probs)
            probabilities = list(probs_tnsr.numpy()[0])
            pred_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    acc = 100 * correct_count / all_count # model accuracy
    return acc, correct_count, all_count

def try_model(model_path, device):
    model = torch.load(model_path)

    real_test_loader = data_loader.get_real_image_loader()
    
    acc, correct_count, all_count = test(model, real_test_loader, device)

    print("Accuracy: " + str(acc))
# train script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import ssl
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import VisionTransformer
from matplotlib import pyplot as plt
import numpy as np
from torch.nn.functional import one_hot
from torch.utils.data import random_split
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
import time

class MixUp:
    def __init__(self, num_classes, alpha=0.2, method=1):
        self.alpha = alpha
        self.method = method
        self.num_classes = num_classes

    def __call__(self, images, labels):
        labels = one_hot(labels, self.num_classes)
        
        if self.method == 1:
            lambd = torch.distributions.Beta(self.alpha, self.alpha).sample((images.size(0),)).to(images.device)
        elif self.method == 2:
            lambd = torch.distributions.Uniform(0.1, 0.4).sample((images.size(0),)).to(images.device)

        #mixing random images
        indices = torch.randperm(images.size(0))

        images1, labels1 = images, labels
        images2, labels2 = images[indices], labels[indices]

        mix_images = lambd.view(-1, 1, 1, 1) * images1 + (1 - lambd).view(-1, 1, 1, 1) * images2
        mix_labels = lambd.view(-1, 1) * labels1 + (1 - lambd).view(-1, 1) * labels2

        return mix_images, mix_labels
    
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    
def calculate_metrics(preds, labels, classes):
    # Initialize the confusion matrix
    confusion_matrix = torch.zeros(len(classes), len(classes))

    # Calculate the confusion matrix
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate precision, recall, and F1-score
    precision = confusion_matrix.diag() / confusion_matrix.sum(1)
    recall = confusion_matrix.diag() / confusion_matrix.sum(0)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate true positives
    true_positives = confusion_matrix.diag()

    return precision, recall, f1_score, true_positives

def train_and_evaluate(trainloader, validationloader, holdoutloader, num_epochs, classes, save_filename, sampling_method):
    t0 = time.time()
    ## vision transformer 
    net = VisionTransformer(image_size=32, patch_size=8, num_layers=6, num_heads=8,
                            hidden_dim=384, mlp_dim=1536, dropout=0.0, num_classes=len(classes)).to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    mixUp = MixUp(num_classes=len(classes), alpha=2, method=sampling_method)
    results_train = []
    losses = []
    ## train
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        print('Epoch {}, sampling method {}'.format(epoch+1, sampling_method))
        net.train()
        for i, data in enumerate(trainloader, 0):

            images, labels = data
            images, labels = mixUp(images, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast():
                all_labels_train = []
                all_predictions_train = []
                outputs = net(images.to(device))
                #compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                # Convert one-hot encoded labels to class indices
                labels_indices = labels.argmax(dim=1).to(device)
                #calculate extra metrics
                all_labels_train.extend(labels_indices.tolist())
                all_predictions_train.extend(predicted.tolist())
                precision, recall, f1_score, true_positives = calculate_metrics(torch.tensor(all_labels_train), torch.tensor(all_predictions_train), classes)
                #calculate accuracy
                correct_train += (predicted == labels_indices).sum().item()
                #calculate loss
                loss = criterion(outputs, labels_indices)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Call scaler.step() instead of optimizer.step()
            scaler.update()  # Update the scaler

            # print statistics
            running_loss += loss.item()
            
        average_loss = running_loss / i
        losses.append(running_loss / i) 
        print('Average loss: %.3f' % average_loss)      
        print('Training done.')
        train_accuracy = 100 * correct_train / total_train
        print('Epoch {}, Training accuracy: {}%'.format(epoch+1, train_accuracy))
        #accuracy results
        results_train.append(train_accuracy) 
        for i, class_name in enumerate(classes):
            print(f"Class: {class_name}")
            print(f"Precision: {precision[i]}")
            print(f"Recall: {recall[i]}")
            print(f"F1-score: {f1_score[i]}")
            print(f"True positives: {true_positives[i]}")
            print()
              
        # evaluation on validation
        net.eval()
        correct_val = 0
        total_val = 0
        results_validation = []
        print("Evaluating validation set")

        with torch.no_grad():
            all_labels_val = []
            all_predictions_val = []
            for data in validationloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels.to(device)).sum().item()
                #collect labels and predictions
                all_labels_val.extend(labels.to(device).tolist())
                all_predictions_val.extend(predicted.tolist())

        val_accuracy = 100 * correct_val / total_val
        print('Epoch {}, Validation accuracy: {}%'.format(epoch+1, val_accuracy))
        results_validation.append(val_accuracy)    # accuracy on validation set
        precision, recall, f1_score, true_positives = calculate_metrics(torch.tensor(all_labels_val), torch.tensor(all_predictions_val), classes)
        for i, class_name in enumerate(classes):
            print(f"Class: {class_name}")
            print(f"Precision: {precision[i]}")
            print(f"Recall: {recall[i]}")
            print(f"F1-score: {f1_score[i]}")
            print(f"True positives: {true_positives[i]}")
            print()
            
        #evaluation on test
        results_test = []
        correct_test = 0
        total_test = 0
        print("Evaluating holdout set")

        with torch.no_grad():
            all_labels_test = []
            all_predictions_test = []
            for data in holdoutloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels.to(device)).sum().item()
                #collect labels and predictions
                all_labels_test.extend(labels.to(device).tolist())
                all_predictions_test.extend(predicted.tolist())

        test_accuracy = 100 * correct_test / total_test
        print('Epoch {}, Testing accuracy: {}%'.format(epoch+1, val_accuracy))
        results_test.append(test_accuracy)    # accuracy on validation set
        precision, recall, f1_score, true_positives = calculate_metrics(torch.tensor(all_labels_test), torch.tensor(all_predictions_test), classes)
        for i, class_name in enumerate(classes):
            print(f"Class: {class_name}")
            print(f"Precision: {precision[i]}")
            print(f"Recall: {recall[i]}")
            print(f"F1-score: {f1_score[i]}")
            print(f"True positives: {true_positives[i]}")
            print()
    
    # After training, plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), results_train, label='Train')
    plt.plot(range(1, num_epochs+1), results_validation, label='Test')
    plt.plot(range(1, num_epochs+1), results_test, label='Train')
    plt.title('Accuracy vs. Epoch sampling method '+ str(sampling_method) )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracy_vs_Epoch_metod_'+ str(sampling_method)+ '.png')
    plt.show()
    
    torch.save(net.state_dict(), save_filename)
    print(f"Model {sampling_method} saved.")
    

if __name__ == '__main__':
    
    #ssl._create_default_https_context = ssl._create_unverified_context

    #if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = GradScaler()
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 36

    #dataset
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # full_dataset = torch.utils.data.ConcatDataset([trainset, testset])

    #do subset 
    from torch.utils.data import Subset
    # Define the datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Get the number of samples in the train and test sets
    num_train_samples = len(trainset)
    num_test_samples = len(testset)

    # Calculate the indices for the quarter of the datasets
    train_indices = torch.randperm(num_train_samples)[:num_train_samples//6]
    test_indices = torch.randperm(num_test_samples)[:num_test_samples//6]

    # Create subsets
    trainset_subset = Subset(trainset, train_indices)
    testset_subset = Subset(testset, test_indices)

    # Concatenate the subsets
    full_dataset_subset = torch.utils.data.ConcatDataset([trainset_subset, testset_subset])

    #split
    #80/20 dev to test
    dev_size = int(0.8 * len(full_dataset_subset))
    test_size = len(full_dataset_subset) - dev_size
    development_set, holdout_test_set  = torch.utils.data.random_split(full_dataset_subset, [dev_size, test_size])
    #90/10 train/ test of dev set
    dev_train_size = int(0.9 * len(development_set))
    dev_test_size = len(development_set) - dev_train_size
    development_train_set, development_test_set  = torch.utils.data.random_split(development_set, [dev_test_size, dev_train_size])
    print(len(full_dataset_subset))
    print(len(development_set))
    print(len(development_train_set))
    print(len(development_test_set))
    print(len(holdout_test_set))

    #dataloaders
    trainloader = torch.utils.data.DataLoader(development_train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validationloader = torch.utils.data.DataLoader(development_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    holdoutloader = torch.utils.data.DataLoader(holdout_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
        #do smapling method 1
    train_and_evaluate(trainloader, validationloader, holdoutloader, 1, classes, 'saved_model_sampling_method1_task3.pt', 1)

    #do sampling method 2
    train_and_evaluate(trainloader, validationloader, holdoutloader,  1, classes, 'saved_model_sampling_method2_task3.pt', 2)

    
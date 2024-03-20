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
    """
    A class used to apply MixUp augmentation to images and labels.

    Attributes:
        num_classes (int): The number of classes in the labels.
        alpha (float): The alpha parameter for the Beta distribution. Default is 0.2.
        method (int): The method to use for sampling lambda. 1 for Beta distribution, 2 for Uniform distribution.

    Methods:
        __call__(images, labels): Applies MixUp augmentation to the images and labels.
    """
    def __init__(self, num_classes, alpha=0.2, method=1):
        self.alpha = alpha
        self.method = method
        self.num_classes = num_classes

    def __call__(self, images, labels):
        """
        Applies MixUp augmentation to the images and labels.

        Args:
            images (Tensor): A tensor of images.
            labels (Tensor): A tensor of labels.

        Returns:
            Tensor, Tensor: The augmented images and labels.
        """
        labels = one_hot(labels, self.num_classes)
        
        if self.method == 1:
            lambd = torch.distributions.Beta(self.alpha, self.alpha).sample((images.size(0),)).to(images.device)
        elif self.method == 2:
            lambd = torch.distributions.Uniform(0, 1).sample((images.size(0),)).to(images.device)

        #mixing random images
        indices = torch.randperm(images.size(0))

        images1, labels1 = images, labels
        images2, labels2 = images[indices], labels[indices]

        mix_images = lambd.view(-1, 1, 1, 1) * images1 + (1 - lambd).view(-1, 1, 1, 1) * images2
        mix_labels = lambd.view(-1, 1) * labels1 + (1 - lambd).view(-1, 1) * labels2

        return mix_images, mix_labels
    
def calculate_metrics(preds, labels, classes, one_hot=False):
    """
    Calculates precision, recall, F1-score, true positives, and the number of images for each class.

    Args:
        preds (Tensor): A tensor of predicted labels.
        labels (Tensor): A tensor of true labels.
        classes (list): A list of class names.

    Returns:
        tuple: A tuple containing five elements:
            - precision (Tensor): A tensor of precision values for each class.
            - recall (Tensor): A tensor of recall values for each class.
            - f1_score (Tensor): A tensor of F1-score values for each class.
            - true_positives (Tensor): A tensor of the number of true positives for each class.
            - num_images (Tensor): A tensor of the number of images for each class.
    """
    confusion_matrix = torch.zeros(len(classes), len(classes))

    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    precision = confusion_matrix.diag() / confusion_matrix.sum(1)
    recall = confusion_matrix.diag() / confusion_matrix.sum(0)
    f1_score = 2 * (precision * recall) / (precision + recall)

    true_positives = confusion_matrix.diag()
    num_images = confusion_matrix.sum(1)

    return precision, recall, f1_score, true_positives, num_images

def train_and_evaluate(trainloader, validationloader, holdoutloader, num_epochs, classes, save_filename, sampling_method):
    t0 = time.time()
    ## vision transformer 
    net = VisionTransformer(image_size=32, patch_size=4, num_layers=6, num_heads=8,
                            hidden_dim=512, mlp_dim=1536, dropout=0.0, num_classes=len(classes)).to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    mixUp = MixUp(num_classes=len(classes), alpha=0.4, method=sampling_method)
    results_train = []
    results_test = []
    results_validation = []
    losses_train = []
    losses_test = []
    losses_validation = []
    ## train
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        print('Epoch {}, sampling method {}'.format(epoch+1, sampling_method))
        net.train()
        all_labels_train = []
        all_predictions_train = []
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = mixUp(images, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast():
                outputs = net(images.to(device))
                #compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                # Convert one-hot encoded labels to class indices
                labels_indices = labels.argmax(dim=1).to(device)
                #calculate extra metrics
                all_labels_train.extend(labels_indices.tolist())
                all_predictions_train.extend(predicted.tolist())
                #calculate accuracy
                correct_train += (predicted == labels_indices).sum().item()
                #calculate loss
                loss = criterion(outputs, labels_indices)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Call scaler.step() instead of optimizer.step()
            scaler.update()  # Update the scaler

            # print statistics
            running_loss += loss.item()
        #extra metrics
        precision, recall, f1_score, true_positives, num_images = calculate_metrics(torch.tensor(all_labels_train), torch.tensor(all_predictions_train), classes)
        #report loss
        average_loss = running_loss / i
        losses_train.append(running_loss / i) 
        #report accuarcy
        train_accuracy = 100 * correct_train / total_train
        num_images_all = len(all_labels_train)
        print('Epoch {}, Training accuracy: {:.2f}%, Average loss:{:.2f}, Images in training:{}'.format(epoch+1, train_accuracy, average_loss, num_images_all))
        results_train.append(train_accuracy) 
        #report other metrics 
        for i, class_name in enumerate(classes):
            print("Class: {}, Images in the class: {}, Precision: {:.2f}%, Recall: {:.2f}%, F1-score: {:.2f}%, True positives: {}".format(class_name, num_images[i], precision[i]*100, recall[i]*100, f1_score[i]*100, true_positives[i]))        
        
        # evaluation on validation
        net.eval()
        correct_val = 0
        total_val = 0
        all_labels_val = []
        all_predictions_val = []
        running_loss = 0.0
        average_loss = 0.0

        with torch.no_grad():
            for data in validationloader:
                images, labels = data
                outputs = net(images.to(device))
                loss = criterion(outputs, labels.to(device))
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels.to(device)).sum().item()
                #collect labels and predictions
                all_labels_val.extend(labels.to(device).tolist())
                all_predictions_val.extend(predicted.tolist())
        #report loss
        average_loss = running_loss / i
        losses_validation.append(running_loss / i) 
        #report accuracy
        val_accuracy = 100 * correct_val / total_val
        num_images_all = len(all_labels_val)
        print('Epoch {}, Validation accuracy: {:.2f}%, Average loss:{:.2f}, Images in validation: {}'.format(epoch+1, val_accuracy, average_loss, num_images_all))
        results_validation.append(val_accuracy)    # accuracy on validation set
        precision, recall, f1_score, true_positives, num_images = calculate_metrics(torch.tensor(all_labels_val), torch.tensor(all_predictions_val), classes)
        for i, class_name in enumerate(classes):
            print("Class: {}, Images in the class: {}, Precision: {:.2f}%, Recall: {:.2f}%, F1-score: {:.2f}%, True positives: {}".format(class_name, num_images[i], precision[i]*100, recall[i]*100, f1_score[i]*100, true_positives[i]))        
    #evaluation on holdout
    average_loss = 0.0    
    correct_test = 0
    total_test = 0
    all_labels_test = []
    all_predictions_test = []
    running_loss = 0.0

    with torch.no_grad():
        for data in holdoutloader:
            images, labels = data
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels.to(device)).sum().item()
            #collect labels and predictions
            all_labels_test.extend(labels.to(device).tolist())
            all_predictions_test.extend(predicted.tolist())
    #report loss
    average_loss = running_loss / i
    losses_test.append(average_loss)
    test_accuracy = 100 * correct_test / total_test
    num_images_all = len(all_labels_test)
    print('Epoch {}, Test accuracy: {:.2f}%, Average loss:{:.2f}, Images in test:{:.2f}'.format(epoch+1, test_accuracy, average_loss, num_images_all))
    results_test.append(test_accuracy)    # accuracy on holdout set
    #extra metrics
    precision, recall, f1_score, true_positives, num_images = calculate_metrics(torch.tensor(all_labels_test), torch.tensor(all_predictions_test), classes)
    for i, class_name in enumerate(classes):
        print("Class: {}, Images in the class: {}, Precision: {:.2f}%, Recall: {:.2f}%, F1-score: {:.2f}%, True positives: {}".format(class_name, num_images[i], precision[i]*100, recall[i]*100, f1_score[i]*100, true_positives[i]))    
    # After training, plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), results_train, label='Train')
    plt.plot(range(1, num_epochs+1), results_validation, label='Validation')
    plt.title('Accuracy vs. Epoch sampling method '+ str(sampling_method) )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracy_vs_Epoch_method_'+ str(sampling_method)+ '.png')
    plt.show()
    
    torch.save(net.state_dict(), save_filename)
    print(f"Model {sampling_method} saved.")
    t1 = time.time()
    print("elapsed time : %.2f seconds" % (t1-t0))
    

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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([trainset, testset])


    #split
    #80/20 dev to test
    dev_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - dev_size
    development_set, holdout_test_set  = torch.utils.data.random_split(full_dataset, [dev_size, test_size])
    #90/10 train/ test of dev set
    dev_train_size = int(0.9 * len(development_set))
    dev_test_size = len(development_set) - dev_train_size
    development_train_set, development_test_set  = torch.utils.data.random_split(development_set, [dev_train_size, dev_test_size])   

    #dataloaders
    trainloader = torch.utils.data.DataLoader(development_train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validationloader = torch.utils.data.DataLoader(development_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    holdoutloader = torch.utils.data.DataLoader(holdout_test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #do smapling method 1
    train_and_evaluate(trainloader, validationloader, holdoutloader, 20, classes, 'saved_model_sampling_method1_task3.pt', 1)

    #do sampling method 2
    train_and_evaluate(trainloader, validationloader, holdoutloader, 20, classes, 'saved_model_sampling_method2_task3.pt', 2)
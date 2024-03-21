
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import ssl
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import VisionTransformer
from torch.nn.functional import one_hot
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
import sys


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
            #sampling uniform from [0,1]
            lambd = torch.distributions.Uniform(0, 1).sample((images.size(0),)).to(images.device)

        #mixing random
        indices = torch.randperm(images.size(0))

        images1, labels1 = images, labels
        images2, labels2 = images[indices], labels[indices]

        mix_images = lambd.view(-1, 1, 1, 1) * images1 + (1 - lambd).view(-1, 1, 1, 1) * images2
        mix_labels = lambd.view(-1, 1) * labels1 + (1 - lambd).view(-1, 1) * labels2

        return mix_images, mix_labels
    
def train_and_evaluate(trainloader, testloader, num_epochs, classes, save_filename, sampling_method):
    """
    Trains and evaluates a model.
    Args:
        trainloader (DataLoader): The DataLoader for the training data.
        testloader (DataLoader): The DataLoader for the test data.
        num_epochs (int): The number of epochs to train for.
        save_filename (str): The filename to save the trained model under.
        sampling_method (str): The method to use for sampling during training.

    Returns:
        dict: A dictionary containing the training and test accuracy for each epoch.
    """
    ## vision transformer 
    net = VisionTransformer(image_size=32, patch_size=4, num_layers=6, num_heads=8,
                            hidden_dim=512, mlp_dim=1536, dropout=0.0, num_classes=len(classes)).to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    mixUp = MixUp(num_classes=len(classes),alpha=0.4, method=sampling_method)
    results_train = []
    results_test = []
    ## train
    print('Sampling method {}'.format(sampling_method))
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        print('Epoch {}'.format(epoch+1))
        net.train()
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

                correct_train += (predicted == labels_indices).sum().item()
                loss = criterion(outputs, labels_indices)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Call scaler.step() instead of optimizer.step()
            scaler.update()  # Update the scaler

            # print statistics
            running_loss += loss.item()
            
        average_loss = running_loss / i
        train_accuracy = 100 * correct_train / total_train
        print('Training accuracy: {}%, Average loss: {:.2f}' .format( train_accuracy, average_loss))
        results_train.append(train_accuracy) 
               
        # evaluation on test
        net.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels.to(device)).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print('Testing accuracy: {}%'.format( test_accuracy))
        results_test.append(test_accuracy)    # save trained model
    
    # 36 images
    images_list = []
    labels_list = []

    dataiter = iter(testloader)
    while len(images_list) < 36:
        images, labels = next(dataiter)
        images_list.extend(images)
        labels_list.extend(labels)
        
    #image predictions
    outputs = net(torch.stack(images_list[:36]).to(device))
    _, predicted = torch.max(outputs.data, 1)
    #image params
    
    grid = Image.new('RGB', (6*32, 6*32))

    for i in range(36):
        image = transforms.ToPILImage()(images_list[i]).convert("RGB")
        grid.paste(image, ((i % 6) * 32, (i // 6) * 32))
        print(f"Image {i+1}: Ground Truth: {classes[labels_list[i]]}, Predicted: {classes[predicted[i]]}")

    grid.save(f"result_{sampling_method}.png")

    print('result.png saved.')
    
    #saving models 
    torch.save(net.state_dict(), save_filename)
    print(f"Model {sampling_method} saved.")

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = GradScaler()
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 36

    #training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # example images
    dataiter = iter(trainloader)
    images, labels = next(dataiter) # note: for pytorch versions (<1.14) use dataiter.next()

    #mix up 
    mixup = MixUp(len(classes), 3)
    images, labels = mixup(images, labels)

    #save images
    im = Image.fromarray((torch.cat(images[:16].split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup.jpg")
    print('mixup.jpg saved.')
    class_indices = labels.argmax(dim=1)
    print('Ground truth labels:' + ' '.join('%5s' % classes[class_indices[j]] for j in range(batch_size)))
        

    #do smapling method 1
    train_and_evaluate(trainloader, testloader, 20, classes,  'saved_model_sampling_method1_task2.pt', 1)

    #do sampling method 2
    train_and_evaluate(trainloader, testloader, 20, classes, 'saved_model_sampling_method2_task2.pt', 2)

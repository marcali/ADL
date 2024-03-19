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
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


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
    
def train_and_evaluate(trainloader, testloader, num_epochs, save_filename, sampling_method):
    ## vision transformer 
    net = VisionTransformer(image_size=32, patch_size=8, num_layers=6, num_heads=8,
                            hidden_dim=384, mlp_dim=1536, dropout=0.0, num_classes=len(classes)).cuda()

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    mixUp = MixUp(num_classes=len(classes),alpha=2,method=sampling_method)
    results_train = []
    results_test = []
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
                outputs = net(images.cuda())
                #compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                # Convert one-hot encoded labels to class indices
                labels_indices = labels.argmax(dim=1).cuda()

                correct_train += (predicted == labels_indices).sum().item()
                loss = criterion(outputs, labels_indices)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Call scaler.step() instead of optimizer.step()
            scaler.update()  # Update the scaler

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
        print('Training done.')
        train_accuracy = 100 * correct_train / total_train
        print('Epoch {}, Training accuracy: {}%'.format(epoch+1, train_accuracy))
        results_train.append(train_accuracy)        
        # evaluation on test
        net.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels.cuda()).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print('Epoch {}, Testing accuracy: {}%'.format(epoch+1, test_accuracy))
        results_test.append(test_accuracy)    # save trained model
    
    # After training, plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), results_train, label='Train')
    plt.plot(range(1, num_epochs+1), results_test, label='Test')
    plt.title('Accuracy vs. Epoch sampling method '+ str(sampling_method) )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracy_vs_Epoch_metod_'+ str(sampling_method)+ '.png')
    plt.show()
    
    # Get some random test images
    #TODO: make sure this generates 36 images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Get predictions for these images
    outputs = net(images.cuda())
    _, predicted = torch.max(outputs.data, 1)

    # Prepare the figure
    fig = plt.figure(figsize=(10, 10))

    # For each image in the batch
    for i in range(batch_size):
        ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
        imshow(images[i])
        ax.set_title(f"GT:{classes[labels[i]]}\nPred:{classes[predicted[i]]}", color=("green" if predicted[i]==labels[i] else "red"))

    # Save the figure
    plt.savefig(f"result_{sampling_method}.png")
    print('result.png saved.')
    
    torch.save(net.state_dict(), save_filename)
    print(f"Model {sampling_method} saved.")



ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':

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

    #TODO: make sure this generates 16 images
    #mix up 
    mixup = MixUp(len(classes), 2)
    images, labels = mixup(images, labels)

    #save images
    im = Image.fromarray((torch.cat(images[:16].split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup.jpg")
    print('mixup.jpg saved.')
    class_indices = labels.argmax(dim=1)
    print('Ground truth labels:' + ' '.join('%5s' % classes[class_indices[j]] for j in range(batch_size)))
        
    #do smapling method 1
    train_and_evaluate(trainloader, testloader, 1, 'saved_model_sampling_method1.pt', 1)

    #do sampling method 2
    train_and_evaluate(trainloader, testloader, 1, 'saved_model_sampling_method2.pt', 2)

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



ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    scaler = GradScaler()
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

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

    #TODO:
    #mix up

    #save images
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup.jpg")
    print('mixup.jpg saved.')
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


    #training
    ## vision transformer 
    net = VisionTransformer(image_size=32, patch_size=2, num_layers=6, num_heads=16,
                            hidden_dim=512, mlp_dim=2048, num_classes=len(classes)).cuda()

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    results_train = []
    results_test = []

    ## train
    for epoch in range(20):  # loop over the dataset multiple times
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #TODO: maybe romove autocast for the final one
            with autocast():
                outputs = net(inputs.cuda())
                #compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels.cuda()).sum().item()
                loss = criterion(outputs, labels.cuda())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Call scaler.step() instead of optimizer.step()
            scaler.update()  # Update the scal
            # loss.backward()
            # optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
            accuracy = 100 * correct_train / total_train
        print('Training done.')
        results_train.append(accuracy)
        print('Accuracy of the network on the train images: {}%'.format(accuracy))
        
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

        print('Accuracy of the network on the test images: {} %'.format(100 * correct_test / total_test))
    # save trained model
    torch.save(net.state_dict(), 'saved_model.pt')
    print('Model saved.')
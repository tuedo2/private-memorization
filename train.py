import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet_gn import ResNetGN18

from opacus import PrivacyEngine

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) # change flag to True to download
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

EPSILON = 10
DELTA = 1 / len(train)
MAX_GRAD_NORM = 1.0

if __name__ == '__main__':

    net = ResNetGN18(10).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    EPOCHS = 40

    privacy_engine = PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM
    )

    
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Loss [{epoch+1}, {i+1}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0
        
        if (epoch+1) % 4 == 0:
            path = f'./eps10/epoch_{epoch+1}.pth'
            torch.save(net.state_dict(), path)

    

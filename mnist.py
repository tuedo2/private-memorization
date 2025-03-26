import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from alexnet import AlexNet

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model_dir = f'publ_alexnet'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0, ))])
train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)

EPOCHS = 10
batch_size=32
criterion = nn.CrossEntropyLoss()

trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print(f'Saving AlexNet')
    
    net = AlexNet(10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
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
        
        
    path = f'./{model_dir}.pth'
    torch.save(net.state_dict(), path)
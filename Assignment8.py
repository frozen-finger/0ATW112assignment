import torch as torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)    #16*28*28
        self.pool1 = nn.MaxPool2d(2, 2)     #16*14*14
        self.conv2 = nn.Conv2d(16, 32, 3)   #32*12*12
        self.pool2 = nn.MaxPool2d(2, 2)     #32*6*6
        self.conv3 = nn.Conv2d(32, 64, 1)   #64*6*6
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)    #32*6*6
        self.conv5 = nn.Conv2d(32, 16, 1, padding=1)    #16*8*8
        self.pool3 = nn.MaxPool2d(2, 2)     #16*4*4#
        self.fc1 = nn.Linear(16*4*4, 120)
        self.norm1 = nn.BatchNorm1d(120)
        self.Re = nn.ReLU()
        self.fc2 = nn.Linear(120, 60)
        self.norm2 = nn.BatchNorm1d(60)
        self.fc3 = nn.Linear(60, 10)


    def forward(self, input):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.pool3(output)
        output = output.view(-1, 16*4*4)
        output = self.fc1(output)
        # output = self.norm1(output)
        # output = self.Re(output)
        output = self.fc2(output)
        # output = self.Re(output)
        # # output = self.norm2(output)
        output = self.fc3(output)
        return output

cnn = Cnn().cuda()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
traindata = torchvision.datasets.CIFAR10(root="./assignment8data", train=True, transform=transform, download=True)
testdata = torchvision.datasets.CIFAR10(root="./assignment8data", train=False, transform=transform, download=True)
trainset = torch.utils.data.DataLoader(traindata, batch_size=4, shuffle=True)
testset = torch.utils.data.DataLoader(traindata, batch_size=4, shuffle=False)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
for epoch in range(10):
    sumloss = 0.0
    for data in trainset:
        x, y = data
        x = x.cuda()
        y = y.cuda()
        output = cnn(x)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        sumloss+=loss
    print("epoch{0} , loss:{1}".format(epoch, sumloss))

#test
correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        x, y = data
        x = x.cuda()
        y = y.cuda()
        pred_y = cnn(x)
        pred_y = torch.argmax(pred_y, 1)
        correct += (pred_y == y).sum().item()
        total += pred_y.size(0)
print("correct/total={}".format((correct/total)))





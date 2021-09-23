from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import torch
import PIL.ImageOps
from PIL import Image

# global variable for progress bar, needs to be global since QThreads update them
global progress
progress = 0


# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.20)
        self.dropout2 = nn.Dropout2d(0.20)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # x represents our data
    # INPUT: forward pass data
    # OUTPUT: softmaxed distribution array
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output


# model structure and definition
global model
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# called each epoch to train the model
# INPUT: Number of Epochs to process
# OUTPUT: NA

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# compare the test set with labels
# INPUT: NA
# OUTPUT: NA


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return (f'Test set: Average loss: {test_loss:.4f},   Accuracy: {correct}/{len(test_loader.dataset)} '
            f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# download mnist from the database
# INPUT: NA
# OUTPUT: NA


def downloadMNIST():

    # MNIST Dataset
    print("Download MNIST")
    global train_datase, train_loader, test_loader, test_dataset
    train_dataset = datasets.MNIST(root='mnist_data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
    # Data Loader (Input Pipeline)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_dataset = datasets.MNIST(root='mnist_data/',
                                  train=False,
                                  transform=transforms.ToTensor())
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

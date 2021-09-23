from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from train import model
import numpy as np
import sys
import time
import torch
import PIL.ImageOps
from PIL import Image, ImageChops

# batch and device information for using the model
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Using {device}\n{"=" * 44}')

# Model structure class
# When a model is loaded, number of layers, settings and output needs to match


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

        # Apply log softmax to x for probability distribution summing to 1
        output = F.log_softmax(x, dim=1)
        return output

# function to classify an image
# INPUT: Model Path, Image Path
# OUTPUT: Probability Distribution Array


def recognise(model_path, image_path):
    global model
    im = Image.open('scripts/images/1/0.jpg')
    immat = im.load()
    (X, Y) = im.size
    m = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            m[x, y] = immat[(x, y)] != (255)
    m = m / np.sum(np.sum(m))

    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    # expected values
    cx = int(np.sum(dx * np.arange(X)))
    cy = int(np.sum(dy * np.arange(Y)))
    figure = plt.figure()

    # shifts the image with calculated center location
    im = ImageChops.offset(im, int(X - (X/2) - cx), int(Y - (Y/2) - cy))
    im.paste((255), (0, 0, int(X - (X/2) - cx), Y))
    im.paste((255), (0, 0, X, int(Y - (Y/2) - cy)))
    im.save('scripts/images/1/0.jpg')

    # extract the hand drawn image in the image folder, and transform
    data_dataset = datasets.ImageFolder(root=image_path,
                                        transform=transforms.Compose([
                                            transforms.Resize(20),
                                            transforms.Grayscale(
                                                num_output_channels=1),
                                            transforms.Pad(
                                                padding=4, fill=0, padding_mode="constant"),
                                            lambda x: transforms.functional.invert(
                                                x),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])]))
    model.load_state_dict(torch.load(model_path), strict=False)
    images, labels = next(iter(data_dataset))

    # classify the image and output the probability distribution
    with torch.no_grad():
        logps = model(images.unsqueeze(0))
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        return probab

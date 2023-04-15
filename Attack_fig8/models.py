import numpy as np
import matplotlib.pyplot as plt
import constants as cons

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import torchvision.datasets as datasets
from torchvision import datasets, models, transforms
from torch.autograd import Variable


# load model
model = models.vgg13(pretrained=True)

# Make sure the input type and the weight type be on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_size=(3, 244, 244)

if cons.SHOW:
    summary(model, input_size)

# Initialize VGG-13 model
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=cons.LEARNING_RATE,
                      momentum=cons.MOMENTUM
                      )


def train(train_loader, train_dataset):
    for epoch in range(cons.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        grads = ''

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()

            # Calculate gradients and store them
            grads = [param.grad.data.view(-1).clone().cpu() \
                     for param in model.parameters() if param.grad is not None]
            break

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch {epoch+1}/{cons.NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")


        # Calculate average gradient for the epoch
        # grads = torch.cat(grads)
        avg_grad = torch.mean(grads).item()
        print(f"Epoch {epoch+1} - Average Gradient:", avg_grad)
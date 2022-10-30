import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/cifar10')))
from cifar10 import CIFAR10
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/cifar10')))
from model import ResNet18

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
cifar10 = CIFAR10()

# Define loaders
train_loader = cifar10.get_train_data_loader(batch_size=128, shuffle=True)

# Define the model
net = ResNet18().to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Train the network
for epoch in range(200):
  net.train()
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = net(images)
    loss = criterion(outputs, labels)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print statistics
    if (i + 1) % 100 == 0:
      print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(epoch + 1, 200, i + 1, len(train_loader), loss.item()))
  # Save the model checkpoint every 10 epochs
  if (epoch + 1) % 50 == 0:
    torch.save(net.state_dict(), f'model_{epoch + 1}.pth')

torch.save(net.state_dict(), 'model.pth')

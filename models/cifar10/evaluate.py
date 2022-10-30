import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/cifar10')))
from cifar10 import CIFAR10
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/cifar10')))
from model import ResNet18
import matplotlib.pyplot as plt

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
cifar10 = CIFAR10()

# Define loaders
test_loader = cifar10.get_test_data_loader(batch_size=128, shuffle=True)

# Define the model
net = ResNet18()
# Load the model
net.load_state_dict(torch.load(
  os.path.abspath(os.path.join(os.path.dirname(__file__), 'model.pth'))))
net.to(device)

# Test the network and show some results
correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy: {}%'.format(100 * correct / total))

# Display some images with their predicted labels, true labels and confidence
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)
fig = plt.figure(figsize=(25, 4))
for idx in range(20):
  ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
  plt.imshow(images[idx].cpu().numpy().transpose((1, 2, 0)))
  # Display the predicted label and the true label and the confidence
  ax.set_title("{} ({}) {:.2f}%".format(cifar10.classes[predicted[idx]], cifar10.classes[labels[idx]],
                100 * torch.softmax(outputs, dim=1)[idx][predicted[idx]].item()),
                color=("green" if predicted[idx] == labels[idx] else "red"))
plt.show()

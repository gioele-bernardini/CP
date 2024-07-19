import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# Define the number of subprocesses to use for data loading,
# the batch size, and validation size
num_workers = 0
batch_size = 64  # Set a realistic batch size
valid_size = 0.2  # Set a realistic validation size

# Convert to float tensor
transform = transforms.ToTensor()

# Choose the training and test datasets
train_data = datasets.MNIST(
  root='data',
  train=True,
  download=True,
  transform=transform,
)

test_data = datasets.MNIST(
  root='data',
  train=False,
  download=True,
  transform=transform,
)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(
  train_data,
  batch_size=batch_size,
  sampler=train_sampler,
  num_workers=num_workers,
)

valid_loader = DataLoader(
  train_data,
  batch_size=batch_size,
  sampler=valid_sampler,
  num_workers=num_workers,
)

test_loader = DataLoader(
  test_data,
  batch_size=batch_size,
  num_workers=num_workers,
)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    hidden_1 = 512
    hidden_2 = 512
    self.fc1 = nn.Linear(28 * 28, hidden_1)
    self.fc2 = nn.Linear(hidden_1, hidden_2)
    self.fc3 = nn.Linear(hidden_2, 10)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # flatten image input
    x = x.view(-1, 28 * 28)
    # add hidden layer, with relu activation function
    x = F.relu(self.fc1(x))
    # add dropout layer
    x = self.dropout(x)
    # add hidden layer, with relu activation function
    x = F.relu(self.fc2(x))
    # add dropout layer
    x = self.dropout(x)
    # add output layer
    x = self.fc3(x)
    return x

# Initialize the model
model = Net()
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 10
valid_loss_min = np.Inf

# Training and validation loop
for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch + 1,
        train_loss,
        valid_loss
    ))

# Testing loop
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
correct_total = 0

model.eval()
for data, target in test_loader:
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    _, pred = torch.max(output, 1)
    correct = pred.eq(target.data.view_as(pred))

    correct_total += correct.sum().item()

    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# Calculate and print avg test loss
test_loss = test_loss / len(test_loader.sampler)
accuracy = 100. * correct_total / len(test_loader.sampler)

print('Test Loss: {:.6f}'.format(test_loss))
print('Test Accuracy (Overall): {:.2f}% ({} / {})'.format(
    accuracy, correct_total, len(test_loader.sampler)
))

# Visualization of the results
# Obtain one batch of images from the test_loader
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.numpy()

output = model(torch.from_numpy(images))
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))
# Ensure to use the minimum length between images, labels, and preds
num_images = min(len(images), len(labels), len(preds), 20)
for i in np.arange(num_images):
    ax = fig.add_subplot(2, num_images // 2, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[i].item()), str(labels[i].item())),
                 color=("green" if preds[i] == labels[i] else "red"))

plt.show()


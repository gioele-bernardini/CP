#!/usr/bin/env python3

# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# --- Data loading and preprocessing ---

# Set parameters for data loading
num_workers = 0  # Number of subprocesses to use for data loading
batch_size = 64  # Batch size for training, validation, and testing
valid_size = 0.2  # Percentage of training data to use for validation

# Define a simple transformation that converts images to PyTorch tensors
transform = transforms.ToTensor()

# Load the MNIST dataset (images of handwritten digits) for training
train_data = datasets.MNIST(
  root='data',  # Directory where data is stored
  train=True,  # Load the training set
  download=True,  # Download the dataset if not already present
  transform=transform,  # Apply the transformation to the data
)

# Load the MNIST dataset for testing
test_data = datasets.MNIST(
  root='data',  # Directory where data is stored
  train=False,  # Load the test set
  download=True,  # Download the dataset if not already present
  transform=transform,  # Apply the transformation to the data
)

# --- Splitting the data for training and validation ---

# Get the number of training samples
num_train = len(train_data)

# Create a list of indices for the training samples
indices = list(range(num_train))

# Shuffle the indices to ensure random sampling
np.random.shuffle(indices)

# Calculate the split point for validation data
split = int(np.floor(valid_size * num_train))

# Split the indices into training and validation sets
train_idx, valid_idx = indices[split:], indices[:split]

# Create samplers for the training and validation sets
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# --- Creating data loaders for training, validation, and testing ---

# DataLoader for the training set with the specified batch size and sampler
train_loader = DataLoader(
  train_data,
  batch_size=batch_size,
  sampler=train_sampler,
  num_workers=num_workers,
)

# DataLoader for the validation set with the specified batch size and sampler
valid_loader = DataLoader(
  train_data,
  batch_size=batch_size,
  sampler=valid_sampler,
  num_workers=num_workers,
)

# DataLoader for the test set with the specified batch size
test_loader = DataLoader(
  test_data,
  batch_size=batch_size,
  num_workers=num_workers,
)

# --- Defining the neural network model ---

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # Define the architecture of the network

    # First hidden layer with 512 neurons
    hidden_1 = 512

    # Second hidden layer with 512 neurons
    hidden_2 = 512

    # Fully connected layer from input (28x28 pixels) to first hidden layer
    self.fc1 = nn.Linear(28 * 28, hidden_1)

    # Fully connected layer from first hidden layer to second hidden layer
    self.fc2 = nn.Linear(hidden_1, hidden_2)

    # Fully connected layer from second hidden layer to output (10 classes for 10 digits)
    self.fc3 = nn.Linear(hidden_2, 10)

    # Dropout layer to prevent overfitting (randomly sets some neurons to zero)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # Flatten the input image (28x28) into a single vector (784 elements)
    x = x.view(-1, 28 * 28)

    # Pass through first hidden layer with ReLU activation and dropout
    x = F.relu(self.fc1(x))
    x = self.dropout(x)

    # Pass through second hidden layer with ReLU activation and dropout
    x = F.relu(self.fc2(x))
    x = self.dropout(x)

    # Pass through output layer (no activation needed as it's used in loss function)
    x = self.fc3(x)
    return x

# Initialize the model
model = Net()
print(model)

# --- Setting up the loss function and optimizer ---

# Use CrossEntropyLoss which combines LogSoftmax and NLLLoss
criterion = nn.CrossEntropyLoss()

# Use Stochastic Gradient Descent (SGD) as the optimizer with a learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# --- Training the model ---

# Number of epochs to train the model
n_epochs = 20

# Track the minimum validation loss to save the best model
valid_loss_min = np.Inf

# Training loop over epochs
for epoch in range(n_epochs):
  train_loss = 0.0  # Initialize training loss for the epoch
  valid_loss = 0.0  # Initialize validation loss for the epoch

  # --- Training phase ---
  model.train()  # Set the model to training mode
  for data, target in train_loader:
    optimizer.zero_grad()  # Clear the gradients
    output = model(data)  # Forward pass: compute predicted outputs
    loss = criterion(output, target)  # Calculate loss
    loss.backward()  # Backward pass: compute gradient of the loss
    optimizer.step()  # Update model parameters
    train_loss += loss.item() * data.size(0)  # Accumulate training loss

  # --- Validation phase ---
  model.eval()  # Set the model to evaluation mode
  for data, target in valid_loader:
    output = model(data)  # Forward pass: compute predicted outputs
    loss = criterion(output, target)  # Calculate loss
    valid_loss += loss.item() * data.size(0)  # Accumulate validation loss

  # Calculate average losses for the epoch
  train_loss = train_loss / len(train_loader.sampler)
  valid_loss = valid_loss / len(valid_loader.sampler)

  # Print training and validation loss for the current epoch
  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    epoch + 1,
    train_loss,
    valid_loss
  ))

  # Example: After running, you'll see output like:
  # Epoch: 1 	Training Loss: 0.640359 	Validation Loss: 0.160715

# --- Testing the model ---

# Initialize test loss and correct predictions counters
test_loss = 0.0
class_correct = list(0. for i in range(10))  # Initialize list to count correct predictions for each class
class_total = list(0. for i in range(10))  # Initialize list to count total predictions for each class
correct_total = 0  # Total correct predictions across all classes

# Evaluate the model on the test set
model.eval()
for data, target in test_loader:
  output = model(data)  # Forward pass: compute predicted outputs
  loss = criterion(output, target)  # Calculate test loss
  test_loss += loss.item() * data.size(0)  # Accumulate test loss
  _, pred = torch.max(output, 1)  # Get the class with the highest score
  correct = pred.eq(target.data.view_as(pred))  # Check if prediction matches the target

  correct_total += correct.sum().item()  # Accumulate total correct predictions

  for i in range(len(target)):
    label = target.data[i]
    class_correct[label] += correct[i].item()  # Count correct predictions for each class
    class_total[label] += 1  # Count total predictions for each class

# Calculate and print average test loss
test_loss = test_loss / len(test_loader.sampler)
accuracy = 100. * correct_total / len(test_loader.sampler)

print('Test Loss: {:.6f}'.format(test_loss))
print('Test Accuracy (Overall): {:.2f}% ({} / {})'.format(
  accuracy, correct_total, len(test_loader.sampler)
))

# Example: After running, you might see output like:
# Test Loss: 0.134567
# Test Accuracy (Overall): 96.50% (9650 / 10000)

# --- Visualizing the results ---

# Obtain one batch of images from the test_loader
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.numpy()  # Convert images to numpy array for visualization

# Pass the images through the model to get predictions
output = model(torch.from_numpy(images))
_, preds = torch.max(output, 1)

# Visualize the first 20 images along with their predicted and actual labels
fig = plt.figure(figsize=(25, 4))
num_images = min(len(images), len(labels), len(preds), 20)
for i in np.arange(num_images):
  ax = fig.add_subplot(2, num_images // 2, i + 1, xticks=[], yticks=[])
  ax.imshow(np.squeeze(images[i]), cmap='gray')
  ax.set_title(
    "{} ({})".format(str(preds[i].item()), str(labels[i].item())),
    color=("green" if preds[i] == labels[i] else "red")
  )

plt.show()

# Example: Green titles indicate correct predictions, red indicates incorrect predictions


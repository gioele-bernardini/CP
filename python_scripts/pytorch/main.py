import numpy as np
import torch
from torchvision import datasets, transforms

# Define the number of subprocesses to use for data loading,
# the batch size, and validation size
num_workers = 0
batch_size = 0
valid_size = 0

# Convert to float tensor
transform = transforms.ToTensor()

# Choose the training and test datasets
train_data = datasets.MNIST(
  root = 'data',
  train = True,
  download = True,
  transform = transform,
)

test_data = datasets.MNIST(
  root = 'data',
  train = False,
  download = True,
  transform = transform,
)

num_train = len(train_data)
indices = list(range(num_train))
np.random_shuffle(indices)

split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]



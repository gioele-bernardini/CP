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


train_data = datasets.MNIST(
  root = 'data',
  train = False,
  download = True,
  transform = transform,
)
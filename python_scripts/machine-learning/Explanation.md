### **1. Importing Necessary Libraries**
   - **`numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `matplotlib.pyplot`, `datasets`, `transforms`, `DataLoader`, `SubsetRandomSampler`:**
     - Import essential libraries for data loading, model definition, training, and visualization.

### **2. Loading and Preprocessing the Data**
   - **Setting Parameters:**
     - `num_workers = 0`: Number of subprocesses for data loading.
     - `batch_size = 64`: Batch size for training, validation, and testing.
     - `valid_size = 0.2`: Percentage of the training set reserved for validation.
   - **Defining the Transformation:**
     - `transform = transforms.ToTensor()`: Converts images to PyTorch tensors.
   - **Loading the MNIST Dataset:**
     - `train_data`: Load the training dataset.
     - `test_data`: Load the test dataset.

### **3. Splitting the Data for Training and Validation**
   - **Splitting Indices:**
     - Shuffle and split the training data indices into training and validation sets.
   - **Creating Samplers:**
     - `train_sampler` and `valid_sampler`: Samplers for selecting data samples for training and validation.

### **4. Creating DataLoaders for Training, Validation, and Testing**
   - **`train_loader`, `valid_loader`, `test_loader`:**
     - DataLoaders to iterate over the data in mini-batches during training, validation, and testing.

### **5. Defining the Neural Network Model**
   - **Creating the `Net` Class:**
     - Define a model with three fully connected layers (512, 512, 10 neurons) and dropout layers to prevent overfitting.
   - **Implementing the Forward Pass:**
     - Flatten the image, perform matrix multiplication, add bias, and apply ReLU activation.

### **6. Setting Up the Loss Function and Optimizer**
   - **Loss Function:**
     - `criterion = nn.CrossEntropyLoss()`: Combines LogSoftmax and Negative Log-Likelihood Loss for multi-class classification.
   - **Optimizer:**
     - `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`: Stochastic Gradient Descent with a learning rate of 0.01 to update the model's weights.

### **7. Preparing for Model Training**
   - **Number of Epochs:**
     - `n_epochs = 20`: The model will be trained for 20 epochs.
   - **Tracking Validation Loss:**
     - `valid_loss_min = np.Inf`: Initialize the minimum validation loss to track the best model during training.


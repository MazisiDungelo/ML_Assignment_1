import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class Model(nn.Module):
    """
    Create a neural model with three layers and use ReLu activation
    """
    def __init__(self, input_size):
        """
        Initialize model with input size
        Define three layers 
        """
        super(Model, self).__init__()
        self.input_ly = nn.Linear(input_size, 64)  # Input layer 
        self.hidden_ly = nn.Linear(64, 32)       # Hidden layer 
        self.output_ly = nn.Linear(32, 10)        # output layer 

    
    def forward(self, feature):
        """"
        Define forward pass of the model
        """
        feature = torch.relu(self.input_ly(feature))
        feature = torch.relu(self.hidden_ly(feature))
        feature = self.output_ly(feature)

        return feature
# Retrieve the directory of the data
DATA_DIR = "MNIST_data"
download_dataset = False

#  Download the train data and test data and prepare them
train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset,transform=ToTensor())
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=ToTensor())

# Convert train and test data and labels into floating point tensors
train_data = train_mnist.data.float()
train_labels = train_mnist.targets
test_data = test_mnist.data.float()
test_labels = test_mnist.targets

# Create validation set from the training set
indices = np.random.choice(train_data.shape[0], test_data.shape[0], replace=False)
val_data = train_data[indices]
val_labels = train_labels[indices]
# Deleted the selected data for validated from training set
train_data = np.delete(train_data, indices, axis=0)
train_labels = np.delete(train_labels, indices, axis=0)

# Flatten the images
train_data = train_data.reshape(-1, 28*28)
test_data = test_data.reshape(-1, 28*28)
val_data = val_data.reshape(-1, 28*28)

# Create Pytorch datasets and dataloaders for trasining, validation and testing
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_data = torch.utils.data.TensorDataset(test_data, test_labels)
val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# Define hyperparameters for training
num_epochs = 30
learning_rate = 0.001
input_size = 28*28

model = Model(input_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)

def train(model, train_dataset, criterion, optimizer, epochs):
    """
    Takes model as parameter and train it using dataset and other hyperparameters
    """
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))
                # Compute validation accuracy
                correct = 0
                total = 0
                for images, labels in val_loader:
                    outputs = model(images)
                    _,predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct.item() / total

                print('Validation Accuracy: {} %'.format(accuracy))
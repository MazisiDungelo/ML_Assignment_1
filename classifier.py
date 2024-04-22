import torch
import torch.nn as nn

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
import torch
import torch.nn as nn

"""
Program Name: model.py

Author: DJ Drohan

Student Number: C21315413

Date: 01/03/25

Program Description: 

A Convolutional Neural Network model designed for emotion detection.

- Uses batch normalization to stabilize training.
"""

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        # Initialize the parent class (nn.Module)
        super(CNNModel, self).__init__()

        # Define the convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Convolution layer 1
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for conv1

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Convolution layer 2
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for conv2

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Convolution layer 3
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization for conv3

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Convolution layer 4
        self.bn4 = nn.BatchNorm2d(512)  # Batch normalization for conv4

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # Convolution layer 5 (additional layer)
        self.bn5 = nn.BatchNorm2d(1024)  # Batch normalization for conv5

        # Max pooling layer with kernel size 2 and stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive average pooling to resize output to (4, 4) size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers with dropout to reduce overfitting
        self.fc1 = nn.Linear(1024 * 4 * 4, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Third fully connected layer
        self.fc4 = nn.Linear(128, num_classes)  # Output layer for classification

        # Dropout layers with different probabilities
        self.dropout1 = nn.Dropout(0.3)  # Dropout after first fully connected layer
        self.dropout2 = nn.Dropout(0.4)  # Dropout after second fully connected layer

    def forward(self, x):
        # Apply convolutional layers with batch normalization and activation (ReLU)
        x = self.bn1(torch.relu(self.conv1(x)))  # First convolution + batch normalization + ReLU
        x = self.bn2(torch.relu(self.conv2(x)))  # Second convolution + batch normalization + ReLU
        x = self.pool(x)  # Max pooling after the first two convolutions

        x = self.bn3(torch.relu(self.conv3(x)))  # Third convolution + batch normalization + ReLU
        x = self.bn4(torch.relu(self.conv4(x)))  # Fourth convolution + batch normalization + ReLU
        x = self.pool(x)  # Max pooling after the next two convolutions

        x = self.bn5(torch.relu(self.conv5(x)))  # Fifth convolution + batch normalization + ReLU (no pooling)
        x = self.adaptive_pool(x)  # Adaptive average pooling to reduce the feature map size

        # Flatten the feature map to a 1D tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with dropout for regularization
        x = self.dropout1(torch.relu(self.fc1(x)))  # First fully connected layer with dropout
        x = self.dropout2(torch.relu(self.fc2(x)))  # Second fully connected layer with dropout
        x = torch.relu(self.fc3(x))  # Third fully connected layer without dropout
        x = self.fc4(x)  # Output layer for classification (no activation here)

        return x  # Return the output

# Initialize weights for convolutional and linear layers
def init_weights(m):
    # Initialize weights for convolutional layers using He initialization
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming initialization for Conv2d
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Set bias to 0 if it exists

    # Initialize weights for linear layers using Xavier initialization
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier initialization for Linear layers
        nn.init.constant_(m.bias, 0)  # Set bias to 0 for Linear layers

    # Initialize Batch Normalization layers to have a weight of 1 and bias of 0
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)  # Set batch norm weight to 1
        nn.init.constant_(m.bias, 0)  # Set batch norm bias to 0

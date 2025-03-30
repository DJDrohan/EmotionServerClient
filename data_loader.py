# data_loader.py
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from transform import transform  # Import the transformation function from the transform.py script

"""
Program Name: dataloader.py

Author: DJ Drohan

Student Number: C21315413

Date: 01/03/25

Program Description: 

A program that loads in a given dataset for the Emotion Detection model to use for training and testing.

- Declares both training and testing datasets using the folder names provided.
- Transforms the images in the datasets into tensors using the transformation functions from transform.py.
- Creates batches of 64 items for both training and testing datasets.
- Shuffles the training and testing data to ensure randomness and avoid bias during training and testing.

"""

# Define the path to the dataset directory
dataset_dir = r"shortened kaggle emotion data AHNSS"  # Path where the dataset is located

# Load the train and test datasets

# ImageFolder automatically assigns labels based on the subfolder names in 'train' and 'test' directories.
# For example, if there are subfolders 'angry', 'happy', etc., ImageFolder will assign labels accordingly.

# Apply the transformation (from transform.py) to the images in the dataset
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'test'), transform=transform)

# DataLoader for batching and shuffling data
# Creates batches of 64 samples and shuffles the data for both training and testing datasets to avoid any bias

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # For training data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)  # For testing data

from collections import defaultdict

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

# Import CNN model and its weight initialization function
from model import CNNModel, init_weights
# Import the train dataset for emotion detection
from data_loader import train_dataset

"""
Program Name: train_eval.py

Author: DJ Drohan

Student Number: C21315413

Date: 01/03/25

Program Description: 

A program that trains and evaluates a model using a Convolutional Neural Network
present in model.py.

- Trains the model with the loaded Kaggle FER Facial Emotion Dataset.
- Runs for a specified number of epochs as sent by the main_modelmake.py.
- Evaluates the accuracy of the model by comparing it to the testing dataset.
- Uses CUDA-based GPU if available for faster training.

"""

# Check if a CUDA-capable GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, optimizer, and learning rate scheduler
num_classes = len(train_dataset.classes)  # Get the number of emotion classes
model = CNNModel(num_classes).to(device)  # Instantiate the model and move it to the selected device (CPU/GPU)
model.apply(init_weights)  # Apply custom weight initialization to the model

# Compute class weights based on dataset distribution (to handle class imbalance)
class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2, 3, 4]), y=np.array(train_dataset.targets))
class_weights[3] *= 1.5  # Increase the weight of 'sadness' class by 50% due to prior difficulties learning this class
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Convert to tensor and move to the correct device

# Loss function with weighted classes and label smoothing to improve generalization
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

# Optimizer: Adam with a learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler: Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# Training function to train the model over multiple epochs
def train_model(model, train_loader, val_loader=None, epochs=100, save_every=10, start_epoch=1):
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop over batches
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the selected device

            optimizer.zero_grad()  # Zero the gradients of the optimizer
            outputs = model(images)  # Perform a forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Perform backpropagation
            optimizer.step()  # Update model weights
            running_loss += loss.item()  # Accumulate the loss for the epoch

            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class with the highest probability
            total_train += labels.size(0)  # Increment total samples processed
            correct_train += (predicted == labels).sum().item()  # Count correct predictions

        # Update the learning rate using the scheduler
        scheduler.step()

        # Calculate and print training accuracy
        train_accuracy = 100 * correct_train / total_train
        print(f'Epoch [{epoch}/{start_epoch + epochs - 1}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Save the model every `save_every` epochs
        if epoch % save_every == 0:
            model_filename = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_filename)  # Save the model's state dict
            print(f"Model saved as {model_filename}")

# Function to evaluate the model on the test dataset
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # No gradients are needed for evaluation
    with torch.no_grad():
        # Loop through test data to evaluate performance
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the selected device
            outputs = model(images)  # Perform a forward pass through the model
            _, predicted = torch.max(outputs, 1)  # Get predicted class

            total += labels.size(0)  # Increment total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

            # Track accuracy per class
            for i in range(len(labels)):
                class_total[labels[i].item()] += 1
                if predicted[i] == labels[i]:
                    class_correct[labels[i].item()] += 1

    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'Total Accuracy: {accuracy:.2f}%')

    # Print accuracy per class (emotion)
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]
    for i, emotion in enumerate(emotions):
        if class_total[i] > 0:  # Only print accuracy if the class has samples
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{emotion.capitalize()} Accuracy: {class_acc:.2f}%')

    return accuracy  # Return the overall accuracy of the model

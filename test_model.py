import torch
from model import CNNModel  # Import the CNN model architecture
from data_loader import test_loader  # Import the test data loader

"""
Program Name: test_model.py

Author: DJ Drohan

Date: 20/03/25

Program Description:
This script loads a trained emotion detection model, evaluates it on the test dataset,
and prints the accuracy and per-class performance for each emotion.
"""

# Set the path to the saved model file
MODEL_PATH = "models/67e 76p/best_emotion_cnn.pth"

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise CPU
num_classes = 5  # Number of emotion classes, update if necessary
model = CNNModel(num_classes).to(device)  # Instantiate the model and send it to the appropriate device (CPU/GPU)

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load the weights into the model
model.eval()  # Set the model to evaluation mode (important for inference)

print(f"Loaded model from: {MODEL_PATH}")

# Initialize lists to store accuracies, predictions, and labels
accuracies = []  # List to store accuracy for each run
all_predictions = []  # List to store all predictions
all_labels = []  # List to store all true labels

# Evaluate the model 10 times to calculate the accuracy over multiple runs
for i in range(10):
    # Initialize counters for each run
    correct = 0
    total = 0
    predictions = []  # List to collect predictions for this run
    labels = []  # List to collect true labels for this run

    # Loop through the test dataset
    for inputs, target_labels in test_loader:
        inputs, target_labels = inputs.to(device), target_labels.to(device)  # Move inputs and labels to device
        outputs = model(inputs)  # Get the model's outputs
        _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest output probability

        predictions.extend(predicted.cpu().numpy())  # Store the predictions for this batch
        labels.extend(target_labels.cpu().numpy())  # Store the true labels for this batch

        total += target_labels.size(0)  # Increment total count
        correct += (predicted == target_labels).sum().item()  # Count correct predictions

    # Calculate accuracy for this run
    accuracy = (correct / total) * 100
    accuracies.append(accuracy)  # Store the accuracy for this run

    # Store all predictions and labels for further analysis
    all_predictions.extend(predictions)
    all_labels.extend(labels)

    print(f"Run {i + 1}: Accuracy = {accuracy:.2f}%")

# Calculate and print the average accuracy over all 10 runs
average_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy over 10 runs: {average_accuracy:.2f}%")

# Initialize dictionaries to store correct predictions and total count for each class
class_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Correct predictions for each class (5 classes)
class_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Total instances for each class

# Calculate per-class accuracy by comparing predictions with true labels
for pred, label in zip(all_predictions, all_labels):
    class_total[label] += 1  # Increment total count for the class
    if pred == label:  # If prediction matches true label, increment correct count for the class
        class_correct[label] += 1

# Print the per-class accuracy for each emotion
for class_id in range(num_classes):
    if class_total[class_id] > 0:  # Ensure there are instances of the class
        class_accuracy = (class_correct[class_id] / class_total[class_id]) * 100  # Calculate accuracy
        emotion = ["Angry", "Happy", "Neutral", "Sad", "Surprised"][class_id]  # Emotion names for classes
        print(f"Accuracy for {emotion}: {class_accuracy:.2f}%")

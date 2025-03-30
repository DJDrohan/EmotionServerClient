# main_modelmake.py
import torch
import os
from datetime import datetime
from train_eval import model,train_model, evaluate_model
from data_loader import train_loader, test_loader

"""
Program Name:main_modelmake.py

Author DJ Drohan

Student Number:C21315413

Date:

Program Description: 

A program that compiles an emotion detection model using components from the other model related scripts

train_eval.py
data_loader.py

transform.py (via dataloader)
model.py (via train_eval)

The max amount epochs of training are declared at first

call to model training function

call to model evaluate function

get current timestamp

name model with timestamp

save to models directory

create models directory if it doesnt exist already

make model path

save model with torch using model path and filename

"""
# Set the number of epochs for training
num_epochs = 200
checkpoint_interval = 10  # Save model every 10 epochs

# Create directories if they don't exist
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

#makes a log.txt of the models creation with a timestamped filename

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
accuracy_filename = f'accuracy_log_{current_time}.txt'
accuracy_log_path = os.path.join(model_dir, accuracy_filename)


"""

Verification of what training data and test data is being fed to model 
using class id 0-4 representing the emotions in alphabetical order

data_iter = iter(train_loader)
images, labels = next(data_iter)
print("Sample training labels:", labels[:10])  # Print first 10 labels

data_iter = iter(test_loader)
images, labels = next(data_iter)
print("Sample test labels:", labels[:10])

from collections import Counter

train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]

print("Training Label Distribution:", Counter(train_labels))
print("Test Label Distribution:", Counter(test_labels))

# Fetch a batch of test data
test_iter = iter(test_loader)
test_images, test_labels = next(test_iter)

# Print actual labels from test loader
print("Sample test labels from DataLoader:", test_labels[:10])
print(train_dataset.classes)

"""
#Open accuracy log file
with open(accuracy_log_path, 'a') as log_file:
    log_file.write("Epoch, Accuracy\n")

    best_accuracy = 0.0  # Track best model accuracy

    for epoch in range(1, num_epochs + 1):

        #call to model trainer and evaluator
        train_model(model, train_loader, val_loader=test_loader, epochs=1, save_every=checkpoint_interval)
        accuracy = evaluate_model(model, test_loader)

        # Log accuracy appending
        log_file.write(f"{epoch}, {accuracy:.2f}\n")
        log_file.flush()

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(model_dir, "best_emotion_cnn.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path} (Accuracy: {accuracy:.2f}%)")

        # Save checkpoint every 10 epochs
        if epoch % checkpoint_interval == 0:
            #timestamped filename with amount of epochs the pth went through
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f'emotion_cnn_model_epoch{epoch}_{current_time}.pth'
            checkpoint_path = os.path.join(model_dir, checkpoint_filename)

            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

#Message after 200 epochs
print(f"Training complete. Accuracy log saved to '{accuracy_log_path}'")

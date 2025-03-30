# transform.py

from torchvision import transforms

"""
Program Name:transform.py

Author DJ Drohan

Student Number:C21315413

Date: 01/03/25

Program Description: 

A program that transform dataset images into 48x48 tensors for model training/evaluation

Applies additional changes such as a horizontal flip and random rotations upto 10 degrees 
to add robustness to training

"""
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),  # Random zoom in/out
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images randomly
    transforms.RandomRotation(15),  # Increase rotation angle to 15 degrees
   #transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust brightness/contrast
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight shifts (translation)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for better convergence
])

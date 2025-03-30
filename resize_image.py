import cv2
import numpy as np


"""
Program Name: resize_image.py

Author: DJ Drohan

Student Number: C21315413

Date: 01/03/25

Program Description:

This script resizes and pads an image to fit within a specified target width and height while maintaining its aspect ratio. 
It also provides an option to choose the background color for padding if the image doesn't perfectly fit the target dimensions. 
The function accepts the source image and processes it with specified resizing parameters.
"""

def resize_and_pad(img, target_width=640, target_height=480, background_color=(255, 255, 255), interpolation=cv2.INTER_LINEAR):
    """
    Resizes the input image to fit within the target dimensions while maintaining the aspect ratio. 
    It then pads the resized image with a specified background color to ensure it fits the exact target size.

    Arguments:
        img (numpy.ndarray): Source image to be resized and padded.
        target_width (int): Target width in pixels (default is 640).
        target_height (int): Target height in pixels (default is 480).
        background_color (tuple): Background color used for padding (default is white (255, 255, 255)).
        interpolation (int): Interpolation method used for resizing (default is cv2.INTER_LINEAR).

    Returns:
        numpy.ndarray: A new image of the specified target dimensions (width x height) with the resized and padded image.
    
    Raises:
        ValueError: If the input image is not a valid numpy ndarray.
    """
    
    # Check if the image is valid
    if img is None or not isinstance(img, np.ndarray): 
        raise ValueError("Input must be a valid numpy ndarray image.")

    # Calculate scaling factors for width and height to maintain aspect ratio
    scale_width = target_width / img.shape[1]
    scale_height = target_height / img.shape[0]
    scale = min(scale_width, scale_height)  # Use the smaller scale factor to maintain aspect ratio

    # Calculate new dimensions after resizing
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)

    # Resize the image with the chosen interpolation method
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    # Create a blank canvas with the target dimensions and the specified background color
    canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)

    # Calculate offsets to center the resized image on the canvas
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image onto the canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

    # Return the final image (resized and padded)
    return canvas

# EmotionLabel.py

import cv2

"""
Program Name: EmotionLabel.py

Author: DJ Drohan

Student Number: C21315413

Date: 01/03/25

Program Description:

A function that draws the detected emotions label beside the processed face frame.
This helps in displaying the emotion label on the frame with a border around it to ensure readability.
"""

def draw_text_with_border(image, text, position):
    """
    Draws text on an image with a black border around it for better readability.

    Arguments:
        image: The image to draw text on.
        text: The text (emotion label) to draw.
        position: (x, y) coordinates for where to place the text on the image.

    Returns:
        None: This function modifies the image in place and does not return anything.
    """

    # Unpack the x, y position where the text will be drawn
    x, y = position

    # Define the color for the border (black, as it's typically used for text legibility)
    border_colour = (0, 0, 0)
    colour = (0,0,255)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 

    # Draw the black border around the text to make it more readable (by drawing it in multiple directions)
    cv2.putText(image, text, (x - 1, y - 1), font, font_scale, border_colour, thickness)  # Top-left
    cv2.putText(image, text, (x + 1, y - 1), font, font_scale, border_colour, thickness)  # Top-right
    cv2.putText(image, text, (x - 1, y + 1), font, font_scale, border_colour, thickness)  # Bottom-left
    cv2.putText(image, text, (x + 1, y + 1), font, font_scale, border_colour, thickness)  # Bottom-right
    cv2.putText(image, text, (x, y - 1), font, font_scale, border_colour, thickness)  # Above
    cv2.putText(image, text, (x, y + 1), font, font_scale, border_colour, thickness)  # Below
    cv2.putText(image, text, (x - 1, y), font, font_scale, border_colour, thickness)  # Left
    cv2.putText(image, text, (x + 1, y), font, font_scale, border_colour, thickness)  # Right

    # Now, draw the main text in the original color on top of the border
    cv2.putText(image, text, position, font, font_scale, colour, thickness)

import cv2
import torch
from torchvision import transforms
from model import CNNModel
from data_loader import train_dataset


# Access the camera if the user is logged in
def real_time_emotion_detection():

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Load the pre-trained emotion model
    num_classes = len(train_dataset.classes)
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load("models/67e 76p ANGRY IS THE NEW FEAR/best_emotion_cnn.pth", map_location=device))
    model.eval()
    model = model.to(device)

    # Define emotion labels based on the dataset classes
    emotion_mapping = train_dataset.classes

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



    # Transform passed detected face into a 48x48 tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    # Run the actual emotion detection function
    def run_emotion_detection(model, emotion_mapping):
        # Initialize video capture from the default camera (camera index 0)
        cap = cv2.VideoCapture(0)
        while True:
            # Read a frame from the video capture
            ret, frame = cap.read()
            if not ret:
                break  # Exit if the frame was not captured successfully


            # Convert the frame to grayscale, as the emotion detection model likely expects this format
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces within the grayscale frame using a pre-trained face cascade classifier
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Loop through each detected face in the frame
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face in the color green
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the region of interest (ROI) corresponding to the face
                face_roi = gray_frame[y:y + h, x:x + w]

                # Resize the face ROI to 48x48 pixels, the size expected by the emotion model
                face_img = cv2.resize(face_roi, (48, 48))

                # Transform and format the face image for model input, adding a batch dimension and transferring it to the device (GPU/CPU)
                face_tensor = transform(face_img).unsqueeze(0).to(device)

                # Perform emotion prediction without updating model weights
                with torch.no_grad():
                    output = model(face_tensor)  # Get the raw model output
                    probabilities = torch.softmax(output, dim=1)[0]  # Convert outputs to probabilities
                    predicted_class = torch.argmax(probabilities).item()  # Identify the most probable emotion
                    emotion_label = emotion_mapping[predicted_class]  # Map the predicted class to an emotion label

                # Define initial text position for displaying emotion labels
                text_x = x - 200
                text_y = y

                # Display the predicted emotion label above the detected face
                cv2.putText(frame, f'{emotion_label}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                text_y += 20

                # Display each emotion probability in the list
                for i, (emotion_name, prob) in enumerate(zip(emotion_mapping, probabilities.cpu().numpy())):
                    cv2.putText(frame, f'{emotion_name}: {prob * 100:.2f}%', (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    text_y += 20  # Move down for the next line of text

            # Show the frame with annotations in a window titled 'Real-time Emotion Detection'
            cv2.imshow('Real-time Emotion Detection', frame)

            # Capture key press for user interaction
            key = cv2.waitKey(1) & 0xFF  # Mask with 0xFF for compatibility


            # If the 'q' key is pressed, exit the loop
            if key == ord('q'):
                break

        # Release the video capture resource and close any open display windows
        cap.release()
        cv2.destroyAllWindows()

    # Call the inner function to start emotion detection
    run_emotion_detection(model, emotion_mapping)

# Test by calling the main function directly
if __name__ == "__main__":
    real_time_emotion_detection()

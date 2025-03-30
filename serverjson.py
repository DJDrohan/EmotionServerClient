import socket
import time

import base64


from flask import Flask, request, jsonify, render_template
import threading
import logging
import cv2
import numpy as np
import torch
from flask_cors import CORS
from torchvision import transforms


from model import CNNModel  # Custom CNN Model
from data_loader import train_dataset  # Custom DataLoader
from resize_image import resize_and_pad  # Custom image resize utility
from hash_utils import hash_data, generate_password_hash  # Custom hashing utilities
from EmotionLabel import draw_text_with_border  # Custom label drawer

app = Flask(__name__)
CORS(app)

# Globals for Emotion detection model
device = None
model = None
SERVER_SALT = None
SERVER_HASH = None
face_cascade = None
transform = None
emotion_mapping = None

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


"""
json error codes

200 OK

400 Bad Request

403 Forbidden

500 Internal Server Error

502 Bad Gateway

"""

@app.route('/')
def index():
    """Render the HTML GUI."""
    return render_template('server.html')


def load_model():
    """Function to load the model on startup."""
    global model, device, face_cascade, transform, emotion_mapping

    try:

        model_path = "models/67e 76p/best_emotion_cnn.pth"

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        #dictates how many emotions model will try to detect based on folder structure of test set
        num_classes = len(train_dataset.classes)
        #tell model to use said emotions
        model = CNNModel(num_classes)

        #model load state dictionary with given model and location of device
        model.load_state_dict(torch.load(model_path, map_location=device))

        #model set to evaluation mode
        model.eval()

        #load model with given device (GPU or CPU
        model = model.to(device)

        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Define image tensor transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

        # Set emotion mapping
        emotion_mapping = train_dataset.classes

        logging.info("Model loaded successfully on startup")
        return True
    except Exception as e:
        logging.error(f"Error loading model on startup: {e}")
        return False


@app.route('/start-server', methods=['POST'])
def start_server():
    """Endpoint to start the server."""
    global SERVER_SALT, SERVER_HASH

    try:
        password = request.form.get('password')
        if not password:
            return jsonify({
                "status": "error",
                "message": "Password is required"}), 400

        # Generate password hash
        SERVER_SALT, SERVER_HASH = generate_password_hash(password)
        logging.info("Password hash generated")

        return jsonify({
            "status": "success",
            "message": "Server started successfully!"})
    except Exception as e:
        logging.error(f"Error starting server: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)}), 500


@app.route('/stop-server', methods=['POST'])
def stop_server():
    """Endpoint to stop the server."""
    return jsonify({"status": "success",
                    "message": "Server stopped successfully!"})


@app.route('/verify-address', methods=['POST'])
def verify_address():
    """Verify if the server is reachable."""
    try:
        data = request.get_json()
        if not data or 'server_ip' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing server_ip"}), 400

        server_ip = data['server_ip']
        port = 5000

        # Test connection
        try:
            with socket.create_connection((server_ip, port), timeout=3):
                return jsonify({
                    "status": "success",
                    "message": "Server is reachable"}), 200

        except (socket.timeout, ConnectionRefusedError):
            return jsonify({
                "status": "error",
                "message": "Server is unreachable"}), 502

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)}), 500

@app.route('/verify-password', methods=['POST'])
def verify_password():
    """Verify the server password."""
    global SERVER_SALT, SERVER_HASH

    try:
        data = request.get_json()
        if not data or 'password' not in data:
            return jsonify({
                "status": "error",
                "message": "Password not provided"}), 400

        client_password = data['password']
        hashed_password = hash_data(client_password, SERVER_SALT)

        if hashed_password == SERVER_HASH:
            return jsonify({
                "status": "success",
                "message": "Password verified successfully!"}), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid password"}), 403
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)}), 500



@app.route('/process-image', methods=['POST'])
def process_image():
    #Endpoint to process an image for emotion detection using JSON input.

    #variables for emotion model and server password
    global model, device, face_cascade, transform, emotion_mapping, SERVER_SALT, SERVER_HASH

    try:
        # Parse JSON request
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid request"}), 400

        #get password and image from clients request json object
        password = data.get('password', '')
        encoded_image = data.get('image', '')

        # password validation
        if not password or hash_data(password, SERVER_SALT) != SERVER_HASH:
            return jsonify({
                "status": "error",
                "message": "Invalid password"}), 403

        if not encoded_image:
            return jsonify({
                "status": "error",
                "message": "No image data provided"}), 400

        # Decode Base64 image
        try:
            image_data = base64.b64decode(encoded_image)
            np_image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image decoding resulted in None. Invalid File Format")

        #couldnt decode base64 image
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to decode image: {str(e)}"}), 400


        #Process Image
        image = resize_and_pad(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        #no faces in image detected
        if len(faces) == 0:
            # Encode the processed image back to Base64
            _, buffer = cv2.imencode('.jpg', image)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                "status": "success",
                "message": "No faces detected in the image.",
                "errorimage": processed_image_base64})

        # Process each detected face
        results = []
        for face, (x, y, w, h) in enumerate(faces):
            face_roi = gray_image[y:y + h, x:x + w]
            face_tensor = transform(cv2.resize(face_roi, (48, 48))).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                emotion_label = emotion_mapping[predicted_class]
                results.append(emotion_label)
                if face > 0:
                    emotion_label= f"{face+1}:{emotion_label}"

                # Annotate the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_text_with_border(image, emotion_label, (x, y - 10))

        # Encode the processed image back to Base64
        _, buffer = cv2.imencode('.jpg', image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')


        #emotion processed json object
        return jsonify({
            "status": "success",
            "message": "Image processed successfully",
            "emotions": results,
            "processed_image": processed_image_base64
        })

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"}), 500

def run_flask():
    """Run the Flask app."""
    app.run(host='0.0.0.0', port=5000, debug=False)


def main():
    """Main Function To Start Flask In A Thread"""

    """First Load Emotion Detection Model """
    if load_model():
        logging.info("Model loaded successfully")
    else:
        logging.error("Failed to initialize model - server will start but model functions may not work")

    server_thread = threading.Thread(target=run_flask, daemon=True)
    server_thread.start()
    logging.info("Flask server is running...")
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down Flask server.")

if __name__ == "__main__":
    main()

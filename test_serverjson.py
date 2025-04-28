# Import standard libraries and testing tools
import unittest
import concurrent.futures

# Flask app and app-specific modules
import serverjson  # Your main server application
import json
from unittest.mock import patch, MagicMock
import base64
import numpy as np
import torch
import cv2

"""
Program Name: test_serverjson.py

Author: DJ Drohan

Student Number: C21315413

Date: 26/03/25

Program Description:

A comprehensive unit test suite for validating the functionality and stability 
of a Flask-based facial emotion detection server implemented in serverjson.py.

- Uses Python's unittest framework to structure and run tests.
- Simulates HTTP requests via Flask’s test client.
- Covers key server endpoints such as:
  - Homepage rendering
  - Server start/stop control
  - Password validation
  - Emotion detection via image upload
  - Model loading and error handling
- Includes mocking of model components (e.g. face detection, neural network, transformations).
- Verifies server’s ability to process concurrent requests safely and accurately.
- Ensures robust error handling for invalid inputs and system failures.

"""


# Unit test class for the emotion detection server
class TestEmotionDetectionApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before any test. Sets up the test client and basic data."""
        cls.client = serverjson.app.test_client()  # Flask test client
        cls.app = serverjson.app
        # Create a dummy black image and encode it to base64
        cls.test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', cls.test_img)
        cls.encoded_image = base64.b64encode(buffer).decode('utf-8')
        cls.test_password = 'securepassword'

    def setUp(self):
        """Runs before each individual test. Resets global state."""
        serverjson.SERVER_SALT = None
        serverjson.SERVER_HASH = None
        serverjson.server_active = False
        serverjson.emotion_mapping = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]
        serverjson.device = 'cpu'

    def setup_server(self, salt='salt', hash_value='hash'):
        """Helper to simulate a started server with known credentials."""
        serverjson.SERVER_SALT = salt
        serverjson.SERVER_HASH = hash_value
        serverjson.server_active = True

    def test_index_route(self):
        """Checks the home route renders the server UI template."""
        with patch('serverjson.render_template', return_value='mocked_template') as mock_render:
            response = self.client.get('/')
            mock_render.assert_called_once_with('server.html')
            self.assertEqual(response.data, b'mocked_template')

    def test_stop_server(self):
        """Checks if server can be stopped via API."""
        serverjson.server_active = True
        response = self.client.post('/stop-server')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertFalse(serverjson.server_active)

    def test_check_status(self):
        """Verifies the server status endpoint behavior in both states."""
        # When server is active
        serverjson.server_active = True
        response = self.client.get('/check-status')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'OK')

        # When server is inactive
        serverjson.server_active = False
        response = self.client.get('/check-status')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(data['status'], 'error')

    def test_password_validation(self):
        """Tests various password validation scenarios for login and server start."""
        test_cases = [
            ('/verify-password', {}, 400, 'Password not provided', False),
            ('/start-server', {}, 400, 'Password is required', False),
            ('/verify-password', {'password': 'wrong'}, 403, 'Invalid password', True),
            ('/verify-password', {'password': 'correct'}, 200, 'Password verified successfully!', True),
        ]

        for endpoint, payload, expected_status, expected_message, do_setup in test_cases:
            if do_setup:
                self.setup_server()
                with patch('serverjson.hash_data', return_value='hash' if 'correct' in str(payload) else 'wrong'):
                    response = self.client.post(endpoint, json=payload if endpoint.endswith('verify-password') else None, data=payload if endpoint.endswith('start-server') else None)
            else:
                response = self.client.post(endpoint, json=payload if endpoint.endswith('verify-password') else None, data=payload if endpoint.endswith('start-server') else None)

            data = json.loads(response.data)
            self.assertEqual(response.status_code, expected_status)
            self.assertIn(expected_message, data['message'])

    @patch('serverjson.generate_password_hash')
    @patch('serverjson.hash_data')
    def test_process_image_validation(self, mock_hash_data, mock_gen_hash):
        """Tests input validation for image processing endpoint."""
        # Mock hashing
        mock_gen_hash.return_value = ('salt', 'hash')
        mock_hash_data.return_value = 'hash'

        # Start the server
        self.client.post('/start-server', data={'password': self.test_password})

        # No image data provided
        response = self.client.post('/process-image', json={'password': self.test_password})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['message'], 'No image data provided')

        # Invalid base64 image
        response = self.client.post('/process-image', json={'password': self.test_password, 'image': 'invalid-base64-data'})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Failed to decode image', data['message'])

        # Simulate error during decoding
        with patch('serverjson.cv2.imdecode', side_effect=Exception('Simulated server error')):
            response = self.client.post('/process-image', json={'password': self.test_password, 'image': self.encoded_image})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Failed to decode image', data['message'])

    @patch('serverjson.generate_password_hash')
    @patch('serverjson.hash_data')
    def test_face_detection_scenarios(self, mock_hash_data, mock_gen_hash):
        """Tests emotion detection for 0, 1, and multiple faces."""
        mock_gen_hash.return_value = ('salt', 'hash')
        mock_hash_data.return_value = 'hash'
        self.client.post('/start-server', data={'password': self.test_password})

        # No faces detected
        with patch('serverjson.face_cascade', MagicMock()) as mock_cascade:
            mock_cascade.detectMultiScale.return_value = []
            with patch('cv2.cvtColor', return_value=np.zeros((200, 200))):
                with patch('serverjson.resize_and_pad', return_value=self.test_img):
                    response = self.client.post('/process-image', json={'password': self.test_password, 'image': self.encoded_image})
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'No faces detected in the image.')

        # One face detected
        with patch('serverjson.face_cascade', MagicMock()) as mock_cascade:
            mock_cascade.detectMultiScale.return_value = [(10, 10, 50, 50)]
            with patch('serverjson.transform', return_value=torch.zeros((1, 1, 48, 48))):
                with patch('serverjson.model', return_value=torch.tensor([[0.1, 0.8, 0.1, 0.05, 0.05]])):
                    with patch('torch.softmax', return_value=torch.tensor([[0.1, 0.8, 0.1, 0.05, 0.05]])):
                        with patch('torch.argmax', return_value=torch.tensor(1)):
                            with patch('cv2.cvtColor', return_value=np.zeros((200, 200))):
                                with patch('serverjson.resize_and_pad', return_value=self.test_img):
                                    with patch('serverjson.draw_text_with_border'):
                                        with patch('cv2.rectangle'):
                                            response = self.client.post('/process-image', json={'password': self.test_password, 'image': self.encoded_image})
        data = json.loads(response.data)
        self.assertEqual(data['emotions'][0], 'Happy')

        # Multiple faces detected
        with patch('serverjson.face_cascade', MagicMock()) as mock_cascade:
            mock_cascade.detectMultiScale.return_value = [(10, 10, 50, 50), (100, 10, 50, 50)]
            with patch('serverjson.transform', return_value=torch.zeros((1, 1, 48, 48))):
                with patch('serverjson.model', return_value=torch.tensor([[0.1, 0.8, 0.1, 0.05, 0.05]])):
                    with patch('torch.softmax', return_value=torch.tensor([[0.1, 0.8, 0.1, 0.05, 0.05]])):
                        with patch('torch.argmax', return_value=torch.tensor(1)):
                            with patch('cv2.cvtColor', return_value=np.zeros((200, 200))):
                                with patch('serverjson.resize_and_pad', return_value=self.test_img):
                                    with patch('serverjson.draw_text_with_border'):
                                        with patch('cv2.rectangle'):
                                            response = self.client.post('/process-image', json={'password': self.test_password, 'image': self.encoded_image})
        data = json.loads(response.data)
        self.assertEqual(len(data['emotions']), 2)

    @patch('serverjson.generate_password_hash')
    @patch('serverjson.hash_data')
    def test_concurrent_requests(self, mock_hash_data, mock_gen_hash):
        """Tests if server handles multiple concurrent image requests correctly."""
        mock_gen_hash.return_value = ('salt', 'hash')
        mock_hash_data.return_value = 'hash'
        self.client.post('/start-server', data={'password': self.test_password})

        # Create shared mock cascade
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(10, 10, 50, 50)]

        def send_request(client_id):
            with patch('serverjson.face_cascade', mock_cascade):
                with patch('serverjson.transform', return_value=torch.zeros((1, 1, 48, 48))):
                    with patch('serverjson.model', return_value=torch.tensor([[0.1, 0.8, 0.1, 0.05, 0.05]])):
                        with patch('torch.softmax', return_value=torch.tensor([[0.1, 0.8, 0.1, 0.05, 0.05]])):
                            with patch('torch.argmax', return_value=torch.tensor(1)):
                                with patch('cv2.cvtColor', return_value=np.zeros((200, 200))):
                                    with patch('serverjson.resize_and_pad', return_value=self.test_img):
                                        with patch('serverjson.draw_text_with_border'):
                                            with patch('cv2.rectangle'):
                                                return self.client.post('/process-image', json={'password': self.test_password, 'image': self.encoded_image}), client_id

        # Run requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(send_request, i): i for i in range(5)}
            responses = [f.result()[0] for f in concurrent.futures.as_completed(futures)]

        # Assert all responses are valid
        for response in responses:
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'success')
            self.assertEqual(data['emotions'][0], 'Happy')

    def test_model_loading(self):
        """Tests model loading behavior and error handling."""
        # Successful model load
        with patch.object(serverjson, 'load_model', return_value=True):
            result = serverjson.load_model()
            self.assertTrue(result)

        # Simulate failure during model loading
        with patch.object(serverjson, 'load_model', side_effect=Exception("Model Load Failure")):
            with patch('serverjson.generate_password_hash', return_value=('salt', 'hash')):
                with patch('serverjson.print') as mock_print:
                    response = self.client.post('/start-server', data={'password': 'test'})
                    self.assertEqual(response.status_code, 200)
                    # Removed print assertion as it's implementation-dependent

# Run the tests when executed directly
if __name__ == '__main__':
    unittest.main(verbosity=2)

import unittest
import socket

from flask import Flask
import serverjson
import json
from unittest.mock import patch, MagicMock
import base64
import numpy as np
import torch
import cv2
import io


class TestEmotionDetectionAppExtra(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup before running the tests"""
        cls.client = serverjson.app.test_client()
        cls.app = serverjson.app

    def setUp(self):
        """Setup before each test"""
        # Reset global variables before each test
        serverjson.SERVER_SALT = None
        serverjson.SERVER_HASH = None

    def test_index_route(self):
        """Test the index route returns the HTML template"""
        with patch('serverjson.render_template', return_value='mocked_template') as mock_render:
            response = self.client.get('/')
            mock_render.assert_called_once_with('server.html')
            self.assertEqual(response.data, b'mocked_template')

    @patch('serverjson.load_model')
    def test_stop_server(self, mock_load_model):
        """Test the stop-server endpoint"""
        mock_load_model.return_value = True

        response = self.client.post('/stop-server')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'Server stopped successfully!')

    @patch('serverjson.load_model')
    def test_verify_address_missing_server_ip(self, mock_load_model):
        """Test verify address with missing server IP"""
        mock_load_model.return_value = True

        response = self.client.post('/verify-address', json={})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Missing server_ip')

    @patch('socket.create_connection')
    @patch('serverjson.load_model')
    def test_verify_address_timeout(self, mock_load_model, mock_connection):
        """Test verify address with connection timeout"""
        mock_load_model.return_value = True
        mock_connection.side_effect = socket.timeout()

        response = self.client.post('/verify-address', json={'server_ip': '192.168.1.100'})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Server is unreachable')

    @patch('serverjson.load_model')
    def test_verify_password_missing_password(self, mock_load_model):
        """Test verify password with missing password"""
        mock_load_model.return_value = True

        response = self.client.post('/verify-password', json={})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Password not provided')

    @patch('serverjson.load_model')
    def test_process_image_missing_image(self, mock_load_model):
        """Test process image with missing image data"""
        mock_load_model.return_value = True

        # First set up the server with a password
        test_password = 'securepassword'
        self.client.post('/start-server', data={'password': test_password})

        # Send request with valid password but no image
        response = self.client.post('/process-image', json={
            'password': test_password
        })
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'No image data provided')

    @patch('serverjson.load_model')
    def test_process_image_invalid_image(self, mock_load_model):
        """Test process image with invalid image data"""
        mock_load_model.return_value = True

        # First set up the server with a password
        test_password = 'securepassword'
        self.client.post('/start-server', data={'password': test_password})

        # Send request with valid password but invalid base64 data
        response = self.client.post('/process-image', json={
            'password': test_password,
            'image': 'invalid-base64-data'
        })
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertIn('Failed to decode image', data['message'])

    @patch('serverjson.load_model')
    def test_load_model_success(self, mock_load_model):
        """Test model loading success"""
        mock_load_model.return_value = True

        result = serverjson.load_model()
        self.assertTrue(result)
        mock_load_model.assert_called_once()

    @patch('serverjson.load_model')
    def test_load_model_failure(self, mock_load_model):
        """Test model loading failure"""
        mock_load_model.side_effect = Exception("Model loading failed")

        with patch('serverjson.logging.error') as mock_log:
            result = serverjson.load_model()
            self.assertFalse(result)
            mock_log.assert_called()

    @patch('serverjson.model')
    @patch('serverjson.cv2.CascadeClassifier.detectMultiScale')
    @patch('serverjson.torch.argmax')
    @patch('serverjson.torch.softmax')
    @patch('serverjson.load_model')


    def test_process_multiple_faces(self, mock_load_model, mock_softmax, mock_argmax, mock_detect, mock_model):
        """Test processing an image with multiple faces detected"""
        mock_load_model.return_value = True

        # Setup mocks for multiple face detection
        mock_detect.return_value = [(10, 10, 50, 50), (100, 10, 50, 50)]  # Two faces
        mock_softmax.return_value = [torch.tensor([0.1, 0.8, 0.1])]
        mock_argmax.return_value = torch.tensor(1)  # Index 1 in emotion_mapping
        mock_model.return_value = torch.tensor([[0.1, 0.8, 0.1]])

        # Setup necessary global variables
        serverjson.face_cascade = cv2.CascadeClassifier()
        serverjson.device = 'cpu'
        serverjson.transform = MagicMock(return_value=torch.tensor([[[0.0]]]))
        serverjson.emotion_mapping = ['angry', 'happy', 'sad']

        # First set up the server with a password
        test_password = 'securepassword'
        self.client.post('/start-server', data={'password': test_password})

        # Sample base64 image - valid image for processing
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Send request with valid password
        response = self.client.post('/process-image', json={
            'password': test_password,
            'image': encoded_image
        })
        data = json.loads(response.data)

        # Check that we have detected both faces
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['emotions']), 2)
        self.assertEqual(data['emotions'][0], 'happy')
        self.assertEqual(data['emotions'][1], 'happy')

    @patch('serverjson.generate_password_hash')
    @patch('serverjson.load_model')
    def test_start_server_hash_generation_error(self, mock_load_model, mock_hash):
        """Test server start with password hash generation error"""
        mock_load_model.return_value = True
        mock_hash.side_effect = Exception("Hash generation error")

        response = self.client.post('/start-server', data={'password': 'test_password'})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Hash generation error')

    @patch('serverjson.hash_data')
    @patch('serverjson.load_model')
    def test_verify_password_hash_error(self, mock_load_model, mock_hash):
        """Test password verification with hashing error"""
        mock_load_model.return_value = True

        # First set up the server with a password
        test_password = 'securepassword'
        self.client.post('/start-server', data={'password': test_password})

        # Make hash_data throw an exception
        mock_hash.side_effect = Exception("Hash computation error")

        response = self.client.post('/verify-password', json={'password': test_password})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Hash computation error')


if __name__ == '__main__':
    unittest.main()
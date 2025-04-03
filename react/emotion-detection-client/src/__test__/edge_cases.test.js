import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import App from './App';

// Mock axios before tests
jest.mock('axios');

// Mock axios
jest.mock('axios');

// Mock FileReader same as in previous test file
const mockFileReader = {
  readAsDataURL: jest.fn(),
  onload: null,
  result: 'data:image/jpeg;base64,mockedBase64String'
};

window.FileReader = jest.fn(() => mockFileReader);

const triggerFileReaderLoad = () => {
  mockFileReader.onload && mockFileReader.onload({ target: { result: mockFileReader.result } });
};

describe('App Edge Cases and Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('validates file types and rejects invalid files', async () => {
    // Mock successful verifications
    axios.post
      .mockResolvedValueOnce({ data: { status: 'success' } })
      .mockResolvedValueOnce({ data: { status: 'success' } });
    
    render(<App />);
    
    // Verify server and password
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    fireEvent.click(screen.getByText('Verify Password'));
    await waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Try to upload an invalid file type
    const invalidFile = new File(['dummy content'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByLabelText(/Upload Image Password Required/i, { selector: 'input' });
    userEvent.upload(fileInput, invalidFile);
    
    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/Invalid file type! Please upload a valid image/)).toBeInTheDocument();
    });
    
    // Verify no preview is shown
    expect(screen.queryByAltText('Original')).not.toBeInTheDocument();
    expect(screen.getByText('Upload an image to see preview')).toBeInTheDocument();
  });

  test('shows alerts when trying to process without verification', () => {
    render(<App />);
    
    // Mock window.alert
    const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});
    
    // Try to process without verification
    const processButton = screen.getByText('Process Image');
    fireEvent.click(processButton);
    
    // Check alert was shown
    expect(alertMock).toHaveBeenCalledWith('Please verify the server first!');
    
    // Reset and verify server only
    alertMock.mockClear();
    axios.post.mockResolvedValueOnce({ data: { status: 'success' } });
    fireEvent.click(screen.getByText('Verify Server'));
    
    // Wait for server verification
    waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    // Try to process with server verified but no password
    fireEvent.click(processButton);
    expect(alertMock).toHaveBeenCalledWith('Please verify your password first!');
    
    // Reset alert
    alertMock.mockClear();
    
    // Verify password
    axios.post.mockResolvedValueOnce({ data: { status: 'success' } });
    const passwordInput = screen.getByLabelText(/Server Password:/i);
    fireEvent.change(passwordInput, { target: { value: 'test123' } });
    fireEvent.click(screen.getByText('Verify Password'));
    
    // Wait for password verification
    waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Try to process with no file
    fireEvent.click(processButton);
    expect(alertMock).toHaveBeenCalledWith('Please select an image first!');
    
    // Cleanup
    alertMock.mockRestore();
  });

  test('handles server timeout during verification', async () => {
    // Mock server timeout
    axios.post.mockRejectedValueOnce({
      code: 'ECONNABORTED',
      message: 'timeout of 15000ms exceeded'
    });
    
    render(<App />);
    
    // Try to verify server
    const verifyButton = screen.getByText('Verify Server');
    fireEvent.click(verifyButton);
    
    // Wait for timeout message
    await waitFor(() => {
      expect(screen.getByText(/Server connection timed out/)).toBeInTheDocument();
    });
    
    // Check that verification status is still false
    expect(screen.getByText('Verify Server')).toBeInTheDocument();
  });

  test('handles server error response during verification', async () => {
    // Mock server error response
    axios.post.mockRejectedValueOnce({
      response: {
        status: 500,
        statusText: 'Internal Server Error'
      }
    });
    
    render(<App />);
    
    // Try to verify server
    const verifyButton = screen.getByText('Verify Server');
    fireEvent.click(verifyButton);
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/Server error: 500 - Internal Server Error/)).toBeInTheDocument();
    });
  });

  test('handles network error during verification', async () => {
    // Mock network error with no response
    axios.post.mockRejectedValueOnce({
      message: 'Network Error',
      response: undefined
    });
    
    render(<App />);
    
    // Try to verify server
    const verifyButton = screen.getByText('Verify Server');
    fireEvent.click(verifyButton);
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/Network error. Please check your connection./)).toBeInTheDocument();
    });
  });

  test('handles failed password verification', async () => {
    // Mock successful server verification but failed password
    axios.post
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Server verification success
      .mockResolvedValueOnce({ data: { status: 'error', message: 'Invalid password' } }); // Password verification fail
    
    render(<App />);
    
    // Verify server
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    // Try to verify password
    const passwordInput = screen.getByLabelText(/Server Password:/i);
    fireEvent.change(passwordInput, { target: { value: 'wrongpassword' } });
    fireEvent.click(screen.getByText('Verify Password'));
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/Password verification failed: Invalid password/)).toBeInTheDocument();
    });
    
    // Check that verification status is still false
    expect(screen.getByText('Verify Password')).toBeInTheDocument();
  });

  test('tries to verify password before server is verified', () => {
    render(<App />);
    
    // Mock window.alert
    const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});
    
    // Try to verify password without server verification
    const passwordInput = screen.getByLabelText(/Server Password:/i);
    fireEvent.change(passwordInput, { target: { value: 'testpass' } });
    
    // Password verify button should be disabled
    const verifyPasswordButton = screen.getByText('Verify Password');
    expect(verifyPasswordButton).toBeDisabled();
    
    // Clean up
    alertMock.mockRestore();
  });

  test('full end-to-end workflow with multiple emotions', async () => {
    // Mock all API calls for a complete workflow
    axios.post
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Server verification
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Password verification
      .mockResolvedValueOnce({ // Image processing with multiple emotions
        data: {
          status: 'success',
          message: 'Image processed successfully',
          processed_image: 'processedImageBase64String',
          emotions: ['happy', 'surprise']
        }
      });
    
    render(<App />);
    
    // Complete server verification
    fireEvent.change(screen.getByLabelText(/Server IP Address:/i), { target: { value: '10.0.0.5' } });
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    // Complete password verification
    fireEvent.change(screen.getByLabelText(/Server Password:/i), { target: { value: 'securepass123' } });
    fireEvent.click(screen.getByText('Verify Password'));
    await waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Upload an image
    const file = new File(['image data'], 'photo.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/Upload Image Password Required/i, { selector: 'input' });
    userEvent.upload(fileInput, file);
    triggerFileReaderLoad();
    
    // Process the image
    fireEvent.click(screen.getByText('Process Image'));
    
    // Wait for results
    await waitFor(() => {
      expect(screen.getByText(/Happy, surprise - 😊/)).toBeInTheDocument();
      expect(screen.getByAltText('Processed')).toBeInTheDocument();
      expect(screen.getByText(/Image processed successfully/)).toBeInTheDocument();
    });
    
    // Verify API was called with correct parameters
    expect(axios.post).toHaveBeenLastCalledWith(
      'http://10.0.0.5:5000/process-image',
      expect.objectContaining({
        filename: 'photo.jpg',
        image: 'mockedBase64String',
        password: 'securepass123'
      }),
      expect.any(Object)
    );
    
    // Clear everything
    fireEvent.click(screen.getByText('Clear'));
    expect(screen.queryByAltText('Original')).not.toBeInTheDocument();
    expect(screen.getByText('Upload an image to see preview')).toBeInTheDocument();
    expect(screen.getByText('Processed image will appear here')).toBeInTheDocument();
  });
});
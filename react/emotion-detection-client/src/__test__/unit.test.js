import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import App from './App';

// Mock axios before tests
jest.mock('axios');


// Mock FileReader
const mockFileReader = {
  readAsDataURL: jest.fn(),
  onload: null,
  result: 'data:image/jpeg;base64,mockedBase64String'
};

window.FileReader = jest.fn(() => mockFileReader);

// Helper function to trigger FileReader onload
const triggerFileReaderLoad = () => {
  mockFileReader.onload && mockFileReader.onload({ target: { result: mockFileReader.result } });
};

describe('App Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders initial app state correctly', () => {
    render(<App />);
    
    expect(screen.getByText('Emotion Detection Client')).toBeInTheDocument();
    expect(screen.getByText('Server Connection')).toBeInTheDocument();
    expect(screen.getByText('Image Processing')).toBeInTheDocument();
    expect(screen.getByText('Status Log')).toBeInTheDocument();
    
    // Check initial status messages
    expect(screen.getByText(/This application and the Emotion Server does not store client data/)).toBeInTheDocument();
    expect(screen.getByText(/Application started. Please verify server connection/)).toBeInTheDocument();
  });

  test('toggles between dark and light mode', () => {
    render(<App />);
    
    // App starts in dark mode by default
    expect(document.body.getAttribute('data-theme')).toBe('dark');
    
    // Click the toggle button (in dark mode, it shows "Light Mode")
    fireEvent.click(screen.getByText('☀️ Light Mode'));
    
    // Check if it switched to light mode
    expect(document.body.getAttribute('data-theme')).toBe('light');
    
    // Toggle back to dark mode
    fireEvent.click(screen.getByText('🌙 Dark Mode'));
    expect(document.body.getAttribute('data-theme')).toBe('dark');
  });

  test('successfully verifies server connection', async () => {
    axios.post.mockResolvedValueOnce({
      data: { status: 'success', message: 'Server verified' }
    });
    
    render(<App />);
    
    // Change server IP
    const ipInput = screen.getByLabelText(/Server IP Address:/i);
    fireEvent.change(ipInput, { target: { value: '192.168.1.100' } });
    expect(ipInput.value).toBe('192.168.1.100');
    
    // Click verify button
    const verifyButton = screen.getByText('Verify Server');
    fireEvent.click(verifyButton);
    
    // Check loading state
    expect(screen.getByText('Verifying...')).toBeInTheDocument();
    
    // Wait for verification to complete
    await waitFor(() => {
      expect(screen.getByText('Verified')).toBeInTheDocument();
    });
    
    // Check status message
    expect(screen.getByText(/Server verified successfully/)).toBeInTheDocument();
    
    // Check if the correct API was called
    expect(axios.post).toHaveBeenCalledWith(
      'http://192.168.1.100:5000/verify-address',
      { server_ip: '192.168.1.100' },
      expect.objectContaining({
        headers: { 'Content-Type': 'application/json' },
        timeout: 15000
      })
    );
  });

  test('handles server verification failure', async () => {
    axios.post.mockRejectedValueOnce({ 
      message: 'Network Error',
      code: 'ECONNABORTED'
    });
    
    render(<App />);
    
    // Click verify button
    const verifyButton = screen.getByText('Verify Server');
    fireEvent.click(verifyButton);
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/Server connection timed out/)).toBeInTheDocument();
    });
    
    // Button should be back to normal state
    expect(screen.getByText('Verify Server')).toBeInTheDocument();
  });

  test('successfully verifies password after server connection', async () => {
    // Mock successful server verification
    axios.post.mockResolvedValueOnce({
      data: { status: 'success', message: 'Server verified' }
    });
    
    // Mock successful password verification
    axios.post.mockResolvedValueOnce({
      data: { status: 'success', message: 'Password verified' }
    });
    
    render(<App />);
    
    // Verify server first
    const verifyServerButton = screen.getByText('Verify Server');
    fireEvent.click(verifyServerButton);
    
    await waitFor(() => {
      expect(screen.getByText('Verified')).toBeInTheDocument();
    });
    
    // Enter password
    const passwordInput = screen.getByLabelText(/Server Password:/i);
    fireEvent.change(passwordInput, { target: { value: 'testpassword123' } });
    expect(passwordInput.value).toBe('testpassword123');
    
    // Verify password
    const verifyPasswordButton = screen.getByText('Verify Password');
    fireEvent.click(verifyPasswordButton);
    
    // Check loading state
    expect(screen.getByText('Verifying...')).toBeInTheDocument();
    
    // Wait for verification to complete
    await waitFor(() => {
      expect(screen.getAllByText('Verified')[1]).toBeInTheDocument();
    });
    
    // Check status message
    expect(screen.getByText(/Password verified successfully/)).toBeInTheDocument();
    
    // Check if the correct API was called
    expect(axios.post).toHaveBeenLastCalledWith(
      'http://192.168.0.1:5000/verify-password',
      { password: 'testpassword123' },
      expect.objectContaining({
        headers: { 'Content-Type': 'application/json' },
        timeout: 15000
      })
    );
  });

  test('handles file selection and preview', async () => {
    // Mock successful verifications to enable the file input
    axios.post.mockResolvedValueOnce({
      data: { status: 'success' }
    }).mockResolvedValueOnce({
      data: { status: 'success' }
    });
    
    render(<App />);
    
    // Verify server and password
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    fireEvent.click(screen.getByText('Verify Password'));
    await waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Create a mock file
    const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
    
    // Get file input and upload file
    const fileInput = screen.getByLabelText(/Upload Image Password Required/i, { selector: 'input' });
    userEvent.upload(fileInput, file);
    
    // Trigger FileReader onload event
    triggerFileReaderLoad();
    
    // Check that preview image is displayed
    const imagePreviewContainer = screen.getByAltText('Original');
    expect(imagePreviewContainer).toBeInTheDocument();
    expect(imagePreviewContainer.src).toContain('data:image/jpeg;base64,mockedBase64String');
  });

  test('successfully processes an image', async () => {
    // Mock successful verifications
    axios.post
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Server verification
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Password verification
      .mockResolvedValueOnce({ // Image processing
        data: {
          status: 'success',
          message: 'Image processed successfully',
          processed_image: 'processedImageBase64String',
          emotions: ['happy']
        }
      });
    
    render(<App />);
    
    // Verify server and password
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    fireEvent.click(screen.getByText('Verify Password'));
    await waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Upload a file
    const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/Upload Image Password Required/i, { selector: 'input' });
    userEvent.upload(fileInput, file);
    triggerFileReaderLoad();
    
    // Process the image
    const processButton = screen.getByText('Process Image');
    fireEvent.click(processButton);
    
    // Check loading state
    expect(screen.getByText('Processing...')).toBeInTheDocument();
    
    // Wait for processing to complete
    await waitFor(() => {
      expect(screen.getByText('Process Image')).toBeInTheDocument();
      expect(screen.getByAltText('Processed')).toBeInTheDocument();
      expect(screen.getByText(/Happy - 😊/)).toBeInTheDocument();
      expect(screen.getByText('Image processed successfully')).toBeInTheDocument();
    });
    
    // Check if the correct API was called
    expect(axios.post).toHaveBeenLastCalledWith(
      'http://192.168.0.1:5000/process-image',
      expect.objectContaining({
        filename: 'test.jpg',
        image: 'mockedBase64String',
        password: ''
      }),
      expect.objectContaining({
        headers: { 'Content-Type': 'application/json' },
        timeout: 30000
      })
    );
  });

  test('handles image processing errors', async () => {
    // Mock successful verifications but failed processing
    axios.post
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Server verification
      .mockResolvedValueOnce({ data: { status: 'success' } }) // Password verification
      .mockResolvedValueOnce({ // Image processing error
        data: {
          status: 'error',
          message: 'Failed to detect faces in image',
          errorimage: 'errorImageBase64String'
        }
      });
    
    render(<App />);
    
    // Verify server and password
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    fireEvent.click(screen.getByText('Verify Password'));
    await waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Upload a file
    const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/Upload Image Password Required/i, { selector: 'input' });
    userEvent.upload(fileInput, file);
    triggerFileReaderLoad();
    
    // Process the image
    const processButton = screen.getByText('Process Image');
    fireEvent.click(processButton);
    
    // Wait for processing to complete with error
    await waitFor(() => {
      expect(screen.getByText('Process Image')).toBeInTheDocument();
      expect(screen.getByText(/Failed to detect faces in image/)).toBeInTheDocument();
      expect(screen.getByText(/No emotion detected/)).toBeInTheDocument();
    });
  });

  test('clears data when clear button is clicked', async () => {
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
    
    // Upload a file
    const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
    const fileInput = screen.getByLabelText(/Upload Image Password Required/i, { selector: 'input' });
    userEvent.upload(fileInput, file);
    triggerFileReaderLoad();
    
    // Check that image preview is shown
    expect(screen.getByAltText('Original')).toBeInTheDocument();
    
    // Click clear button
    const clearButton = screen.getByText('Clear');
    fireEvent.click(clearButton);
    
    // Check that preview is cleared
    expect(screen.queryByAltText('Original')).not.toBeInTheDocument();
    expect(screen.getByText('Upload an image to see preview')).toBeInTheDocument();
    expect(screen.getByText('Processed image will appear here')).toBeInTheDocument();
  });

  test('clears status messages when clear status button is clicked', () => {
    render(<App />);
    
    // Check initial status messages exist
    expect(screen.getByText(/This application and the Emotion Server does not store client data/)).toBeInTheDocument();
    
    // Click clear status button
    const clearStatusButton = screen.getByText('Clear Status');
    fireEvent.click(clearStatusButton);
    
    // Check that status messages are cleared
    expect(screen.getByText('Status messages will appear here')).toBeInTheDocument();
    expect(screen.queryByText(/This application and the Emotion Server does not store client data/)).not.toBeInTheDocument();
  });

  test('handles file drag and drop', async () => {
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
    
    // Get the drop zone
    const dropZone = screen.getByText('Drag & drop your image here').closest('div');
    
    // Create a mock file
    const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
    
    // Mock drag events
    fireEvent.dragOver(dropZone);
    expect(dropZone.classList.contains('bg-dark-secondary')).toBe(true);
    
    fireEvent.dragLeave(dropZone);
    expect(dropZone.classList.contains('bg-dark-secondary')).toBe(false);
    
    // Mock drop event
    const dataTransfer = {
      files: [file]
    };
    
    fireEvent.drop(dropZone, { dataTransfer });
    triggerFileReaderLoad();
    
    // Check that preview image is displayed
    expect(screen.getByAltText('Original')).toBeInTheDocument();
  });
});
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import App from './App';

// Mock axios before tests
jest.mock('axios');

// Mock axios
jest.mock('axios');

// Mock FileReader
const mockFileReader = {
  readAsDataURL: jest.fn(),
  onload: null,
  result: 'data:image/jpeg;base64,mockedBase64String'
};

window.FileReader = jest.fn(() => mockFileReader);

// Create a mocked current time for consistent timestamps
const mockDate = new Date('2023-01-01T12:00:00');
jest.spyOn(global, 'Date').mockImplementation(() => mockDate);

describe('Individual Functions and Hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('updateStatus function adds timestamped messages', () => {
    render(<App />);
    
    // Clear existing status messages
    fireEvent.click(screen.getByText('Clear Status'));
    expect(screen.getByText('Status messages will appear here')).toBeInTheDocument();
    
    // Add a custom status message via server verification (which calls updateStatus)
    axios.post.mockResolvedValueOnce({
      data: { status: 'success', message: 'Server verified' }
    });
    
    fireEvent.click(screen.getByText('Verify Server'));
    
    // Check for the timestamped message
    expect(screen.getByText(/\[\d{1,2}:\d{2}:\d{2}\] Attempting to verify server at/)).toBeInTheDocument();
    
    // After verification completes
    waitFor(() => {
      expect(screen.getByText(/\[\d{1,2}:\d{2}:\d{2}\] Server verified successfully/)).toBeInTheDocument();
    });
  });

  test('status div auto-scrolls to bottom on new messages', async () => {
    render(<App />);
    
    // Get the status div reference
    const statusDiv = screen.getByText(/Status messages will appear here/).closest('div');
    
    // Mock the scrollHeight property
    Object.defineProperty(statusDiv, 'scrollHeight', { value: 1000 });
    
    // Add a status message by verifying server
    axios.post.mockResolvedValueOnce({
      data: { status: 'success' }
    });
    
    fireEvent.click(screen.getByText('Verify Server'));
    
    // Fast-forward timers to trigger the setTimeout callback
    act(() => {
      jest.advanceTimersByTime(100);
    });
    
    // Check that scrollTop was set to scrollHeight
    expect(statusDiv.scrollTop).toBe(1000);
  });

  test('useEffect hook sets data-theme attribute correctly', () => {
    const { rerender } = render(<App />);
    
    // Default should be dark mode
    expect(document.body.getAttribute('data-theme')).toBe('dark');
    
    // Toggle dark mode via button
    fireEvent.click(screen.getByText('☀️ Light Mode'));
    
    // Check the data-theme attribute is updated
    expect(document.body.getAttribute('data-theme')).toBe('light');
  });

  test('clearData function resets all image-related state', async () => {
    // Mock successful verifications
    axios.post
      .mockResolvedValueOnce({ data: { status: 'success' } })
      .mockResolvedValueOnce({ data: { status: 'success' } })
      .mockResolvedValueOnce({ 
        data: {
          status: 'success',
          processed_image: 'processedBase64',
          emotions: ['happy'],
          message: 'Success'
        }
      });
    
    render(<App />);
    
    // Complete the verification process
    fireEvent.click(screen.getByText('Verify Server'));
    await waitFor(() => expect(screen.getByText('Verified')).toBeInTheDocument());
    
    fireEvent.click(screen.getByText('Verify Password'));
    await waitFor(() => expect(screen.getAllByText('Verified')[1]).toBeInTheDocument());
    
    // Upload and process an image
    const file = new File(['image data'], 'test.png', { type: 'image/png' });
    const fileInput = screen
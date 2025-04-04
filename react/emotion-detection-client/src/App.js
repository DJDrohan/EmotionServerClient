import React, { useState, useRef, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import axios from 'axios'; // axios for HTTP requests


function App(){
  // State variables
  const [serverIP, setServerIP] = useState('192.168.0.1');
  const [password, setPassword] = useState('');
  const [serverVerified, setServerVerified] = useState(false);
  const [passwordVerified, setPasswordVerified] = useState(false);
  const [statusMessages, setStatusMessages] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState("");
  const [processStatus, setProcessStatus] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const statusRef = useRef(null);
  const fileInputRef = useRef(null);

  const emotionIcon = {
    happy: "😊",
    sad: "😢",
    angry: "😡",
    surprise: "😮",
    neutral: "😐"
   }

  // Toggle dark/light mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Handle dragging file over 
  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
    console.log('Drag over');
    if (!isLoading) {
      event.target.classList.add(darkMode ? 'bg-dark-secondary' : 'bg-gray-100');
    }
  };

  // Handle dragging file out
  const handleDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
    console.log('Drag leave');
    event.target.classList.remove(darkMode ? 'bg-dark-secondary' : 'bg-gray-100');
  };

  // Handle file drop event
  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.target.classList.remove(darkMode ? 'bg-dark-secondary' : 'bg-gray-100');

    console.log('Dropped file');
    const file = event.dataTransfer.files[0];
    if (file) {
      handleFileChange({ target: { files: [file] } }); // Use the existing file handling function
    }
  };

  // Add status message with timestamp
  const updateStatus = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    const newMessage = `[${timestamp}] ${message}`;
    setStatusMessages(prev => [...prev, newMessage]);
    
    // Auto scroll to bottom of status div
    setTimeout(() => {
      if (statusRef.current) {
        statusRef.current.scrollTop = statusRef.current.scrollHeight;
      }
    }, 100);
  };

  // Verify server connection
  const verifyServer = async () => {
    updateStatus(`Attempting to verify server at ${serverIP}...`);
    setIsLoading(true);
  
    try {
      const response = await axios.get(`http://${serverIP}:5000/check-status`, { timeout: 5000 });
  
      // If the server is up and running
      if (response.status === 200) {
        if (response.data.status === 'OK') {
          setServerVerified(true);
          updateStatus('Server is up and running!');
        } else {
          // If status is not 'OK', display the server-provided message
          setServerVerified(false);
          updateStatus(`Server error: ${response.data.message}`);
        }
      }
    } catch (error) {
      // If the error is a server-side issue, display the error as is
      setServerVerified(false);
      
      if (error.response) {
        // 503 Service Unavailable or other error codes
        if (error.response.status === 503) {
          updateStatus(`Server error: ${error.response.data.message}`);
        } else {
          updateStatus(`Server error: ${error.response.status} - ${error.response.statusText}`);
        }
      } else if (error.request) {
        // No response from server (network error, unreachable, etc.)
        updateStatus('Unable to connect to the server. Please check the IP address or network connection.');
      } else {
        // Unexpected errors
        updateStatus(`Unexpected error: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  
  

  // Verify password
  const verifyPassword = async () => {
    if (!serverVerified) {
      alert('Please verify server connection first');
      return;
    }

    if (!password) {
      alert('Please enter a password');
      return;
    }

    updateStatus('Verifying password...');
    setIsLoading(true);
    
    try {
      // Send request directly to the target server
      const response = await axios.post(`http://${serverIP}:5000/verify-password`, 
        { password: password },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 15000 // 15 second timeout
        }
      );

      console.log("Password verification response:", response.data);

      if (response.data && response.data.status === 'success') {
        setPasswordVerified(true);
        updateStatus('Password verified successfully');
      } else {
        setPasswordVerified(false);
        updateStatus(`Password verification failed: ${response.data?.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error("Password verification error:", error);
      setPasswordVerified(false);
      updateStatus(`Error verifying password: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
    if (file && allowedTypes.includes(file.type)) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = (e) => {
        setPreviewImage(e.target.result);
      };
    } else {
      clearData()
      updateStatus('Invalid file type! Please upload a valid image (JPEG, PNG)');
    }
  };
  
  
  // Process the selected image=
  const processImage = async () => {

    //checks if server ip and password were confirmed and if a file was uploaded before doing the request
    if (!serverVerified) {
      alert('Please verify the server first!');
      return;
    }
    if (!passwordVerified) {
      alert('Please verify your password first!');
      return;
    }
    if (!selectedFile) {
      alert('Please select an image first!');
      return;
    }
  
    updateStatus('Processing image...');
    setIsLoading(true);
  
    const reader = new FileReader();
    reader.readAsDataURL(selectedFile);
    reader.onload = async () => {
      const base64Image = reader.result.split(',')[1]; // Extract Base64 part
  
      try {
        const response = await axios.post(
          `http://${serverIP}:5000/process-image`,
          { filename: selectedFile.name, image: base64Image, password }, //Json object for server
          { headers: { "Content-Type": "application/json" }, timeout: 30000 }
        );
  
        const data = response.data;
  
        console.log("Response Data:", data);
  
        if (data.status === "success") {
          if (data.processed_image) {
            //Take Image from json response object and display line 416 allows for base64 decoding
            setProcessedImage(data.processed_image);
            updateStatus(data.message || "Image processed successfully");
            setProcessStatus(`${data.message}`);
          } else {
            updateStatus(`No processed image received from the server. ${data.message || ''}`);
            setProcessStatus(`${data.message}`);
            setProcessedImage(`${data.errorimage}`)
          }
          setDetectedEmotion(data.emotions ? data.emotions.join(', ') : "No emotion detected");
        } else {
          alert(data.message || "Error processing image");
          updateStatus(`Error processing image: ${data.message || "Unknown error."}`);
        }
      } catch (error) {
        console.error("Error in processImage:", error);
        updateStatus(`Error processing image: ${error.message}`);
      } finally {
        setIsLoading(false);
      }
    };
  };

  // Clear the status messages
  const clearStatus = () => {
    setStatusMessages([]);
  };



  // Clear Data
  const clearData = () => {
    setSelectedFile(null);
    setPreviewImage(null);
    setProcessedImage(null);
    setDetectedEmotion("");
    setProcessStatus("");
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // useEffect to have a clean page on initial load. This acts like a "main" function in a python program.
  useEffect(() => {

    clearStatus(); // Call the clearStatus function on page load

    updateStatus("This application and the Emotion Server does not store client data. The server only processes the image temporarily for emotion detection and sends the results back.");

    updateStatus("Application started. Please verify server connection.");

    document.body.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    
  }, [darkMode]);

  return (
<div className={`min-h-screen ${darkMode ? 'bg-dark-main' : 'bg-light-main'} py-4`}>
  <div className="w-full max-w-[900px] mx-auto px-4">
    <div className="container mx-auto max-w-full">
      <div className="flex justify-between items-center mb-4">
        <h1 className={`text-3xl font-bold text-center ${darkMode ? 'text-white' : 'text-gray-800'}`}>Emotion Detection Client</h1>
        
        {/* Dark/Light Mode Toggle */}
        <button 
          onClick={toggleDarkMode} 
          className={`mode-toggle-btn ${darkMode ? 'light-mode-btn' : 'dark-mode-btn'}`}
        >
          {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>
      </div>

        {/* Status Section */}
        <div className="">
        <div className={`${darkMode ? 'bg-dark-card' : 'bg-white'} p-6 rounded-lg shadow-md`} >
          <div className="flex justify-between items-center mb-2">
            <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Status Log</h2>
            <button
              onClick={clearStatus}
              className={`px-3 py-1 ${darkMode ? 'bg-dark-button text-gray-300' : 'bg-gray-200 text-gray-700'} rounded hover:bg-gray-300`}
            >
              Clear Status
            </button>
          </div>
          <div
              ref={statusRef}
              className="px-3 py-2 border rounded bg-gray-50 h-40 overflow-auto font-mono text-sm"
              aria-live="polite"
          >

            {statusMessages.length === 0 ? (
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Status messages will appear here</p>
            ) : (
              statusMessages.map((msg, index) => (
                <div key={index}>{msg}</div>
              ))
            )}
          </div>
        </div>
        </div>

        {/* Server Connection Section */}
        <div className={`${darkMode ? 'bg-dark-card' : 'bg-white'} p-6 rounded-lg shadow-md`}>
          <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Server Connection</h2>
          
          <div>
            <label className={`block ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Server IP Address:</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={serverIP}
                onChange={(e) => setServerIP(e.target.value)}
                className={`flex-grow px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 ${darkMode ? 'bg-dark-input border-dark-border text-white' : ''}`}
                disabled={isLoading}
              />
              <button
                onClick={verifyServer}
                className={`px-4 py-2 rounded font-medium ${
                  serverVerified
                    ? 'success-button'
                    : 'default-button'
                } text-white disabled:opacity-50`}
                disabled={isLoading}
              >
                {isLoading && serverVerified === false ? 'Verifying...' : serverVerified ? 'Verified' : 'Verify Server'}
                
              </button>
            </div>
          </div>

          {/* Password Verification */}
          <div>
            <label className={`block ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Server Password:</label>
            <div className="flex gap-2">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`flex-grow px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 ${darkMode ? 'bg-dark-input border-dark-border text-white' : ''}`}
                disabled={isLoading || !serverVerified}
              />
              
              <button
                onClick={verifyPassword}
                className={`px-4 py-2 rounded font-medium ${
                  passwordVerified
                    ? 'success-button'
                    : 'default-button'
                } text-white disabled:opacity-50`}
                disabled={isLoading || !serverVerified}
              >
                {isLoading && passwordVerified === false ? 'Verifying...' : passwordVerified ? 'Verified' : 'Verify Password'}
              </button>
            </div>
          </div>
        </div>

        {/* Image Upload Section */}
        <div className={`${darkMode ? 'bg-dark-card' : 'bg-white'} p-6 rounded-lg shadow-md`}>
          
          <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Image Processing</h2>
          
          <div>
            <label className={`block ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Upload Image Password Required</label>

            {/*File Drop Zone */}
            <div
              className={`drop-zone flex justify-center items-center border-4 border-dashed ${darkMode ? 'border-green-400 bg-dark-secondary' : 'border-gray-400 bg-gray-100'} p-8 rounded-lg hover:bg-gray-200`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Drag & drop your image here</p>
            </div>
            
            
            {/* Normal File uploader */}

            <div className="flex gap-2 border-t mt-4 pt-4">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className={`flex-grow px-3 py-2 border rounded ${darkMode ? 'bg-dark-input border-dark-border text-white' : ''}`}
                disabled={isLoading || !passwordVerified}
              />
            

              {/*Clear Data*/}
              <button
                onClick={clearData}
                className={`${darkMode ? 'bg-dark-button text-gray-300' : 'bg-gray-300 text-gray-700'} px-4 py-2 rounded hover:bg-gray-400 disabled:opacity-50`}
                disabled={isLoading || !selectedFile}
              >
                Clear
              </button>
            </div>
            
            {/*Process Image button to send a server request*/}
            <button
              onClick={processImage}
              className="w-full success-button text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
              disabled={isLoading || !serverVerified || !passwordVerified || !selectedFile}
            >
              {isLoading 
              ? 'Processing...' 
              : 'Process Image'}
            </button>
          </div>
          </div>

        <div className={`${darkMode ? 'bg-dark-card' : 'bg-white'} p-6 rounded-lg shadow-md image-section`}>
          {/* Uploaded Image */}
          <div>
            <h3 className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} text-lg font-semibold mb-3`}>Original Image</h3>
            <div className={`p-4 ${darkMode ? 'bg-dark-secondary' : 'bg-gray-50'} h-64 flex items-center justify-center overflow-hidden`}>
              {previewImage ? (
                <img
                  src={previewImage}
                  alt="Original"
                  className="object-contain max-w-full max-h-full image-container"
                />
              ) : (
                <p className={`${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Upload an image to see preview</p>
              )}
            </div>
          </div>
          </div>
          

          <div className={`${darkMode ? 'bg-dark-card' : 'bg-white'} p-6 rounded-lg shadow-md image-section`}>
          {/* Processed Image */}
          <div>
            <h3 className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} text-lg font-semibold mb-3`}>Processed Image</h3>
            <div className={`p-4 ${darkMode ? 'bg-dark-secondary' : 'bg-gray-50'} h-64 flex items-center justify-center overflow-hidden`}>
              {processedImage ? (
                <img
                  src={`data:image/jpeg;base64,${processedImage}`}
                  alt="Processed"
                  className="object-contain max-w-full max-h-full image-container"
                />
              ) : (
                <p className={`${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Processed image will appear here</p>
              )}
            </div>
          </div>

            {/*Highest Emotion Detected */}
            <p className="text-2xl font-bold text-blue-600 mt-2">
            {detectedEmotion
            ? detectedEmotion.split(', ').map(emotion => 
              `${emotion.charAt(0).toUpperCase() + emotion.slice(1)} ${emotionIcon[emotion.toLowerCase()] || '🤷‍♂️'}`
            ).join(', ')
            : "[Emotion Here]"}
            </p>
            <p className="text-2xl font-bold text-blue-600 mt-2">{processStatus || "[Processing Status Here]"}</p>
          </div>
         </div>
      </div>        
      </div>
  );
}

export default App;
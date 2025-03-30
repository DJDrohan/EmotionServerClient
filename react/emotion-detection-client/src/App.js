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
  const statusRef = useRef(null);
  const fileInputRef = useRef(null);

  // Handle dragging file over 
  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
    console.log('Drag over');
    if (!isLoading) {
      event.target.classList.add('bg-gray-100');
    }
  };

  // Handle dragging file out
  const handleDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
    console.log('Drag leave');
    event.target.classList.remove('bg-gray-100');
  };

  // Handle file drop event
  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.target.classList.remove('bg-gray-100');

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
      const formData = new FormData();
      formData.append('server_ip', serverIP);
      
      // Send request directly to the target server
      const response = await axios.post(`http://${serverIP}:5000/verify-address`, 
        { server_ip: serverIP },
        {
          headers: { 'Content-Type': 'application/json' },
          timeout: 15000 // 15 second timeout
        }
      );

      console.log("Server verification response:", response.data);

      if (response.data && response.data.status === 'success') {
        setServerVerified(true);
        updateStatus('Server verified successfully');
      } else {
        setServerVerified(false);
        updateStatus(`Server verification failed: ${response.data?.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error("Server verification error:", error);
      setServerVerified(false);
      updateStatus(`Error verifying server: ${error.message}`);
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

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
  
      //using readAsDataURL as Depending on image size btoa and atob would need a for loop to go through the array buffer.

      const reader = new FileReader();
      reader.readAsDataURL(file); // Converts image to Base64
      reader.onload = (e) => {
        const base64Image = e.target.result.split(',')[1]; // Extract Base64 part
        console.log("Base64 Encoded Image:", base64Image);
        setPreviewImage(e.target.result); // Show preview image
        setProcessedImage(null);//reset processed image section
      };
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
    setSelectedFile(null);// Clear Selected File
    setPreviewImage(null); // Clear Image Preview
    setProcessedImage(null); // Clear Processed Image display
    setDetectedEmotion(""); // Clear the detected emotion
    setProcessStatus(""); // Clear the process status
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // useEffect to have a clean page on initial load. This acts like a "main" function in a python program.
  useEffect(() => {

    clearStatus(); // Call the clearStatus function on page load

    updateStatus("Application started. Please verify server connection.");
    
  }, []);

  return (
<div className="min-h-screen bg-[#1e1e2e] py-4">
  <div className="w-full max-w-[900px] mx-auto px-4">
    <div className="container mx-auto max-w-full">
      <h1 className="text-3xl font-bold  text-center text-white">Emotion Detection Client</h1>

        {/* Status Section */}
        <div className="">
        <div className="bg-white p-6 rounded-lg shadow-md" >
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-xl font-semibold">Status Log</h2>
            <button
              onClick={clearStatus}
              className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              Clear Status
            </button>
          </div>
          <div
            ref={statusRef}
            className="px-3 py-2 border rounded bg-gray-50 h-40 overflow-auto font-mono text-sm"
          >
            {statusMessages.length === 0 ? (
              <p className="text-gray-500">Status messages will appear here</p>
            ) : (
              statusMessages.map((msg, index) => (
                <div key={index}>{msg}</div>
              ))
            )}
          </div>
        </div>
        </div>

        {/* Server Connection Section */}
        <div className="bg-white p-6 rounded-lg shadow-md ">
          <h2 className="text-xl font-semibold">Server Connection</h2>
          
          <div>
            <label className="block text-gray-700 mb-2">Server IP Address:</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={serverIP}
                onChange={(e) => setServerIP(e.target.value)}
                className="flex-grow px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            <label className="block text-gray-700 mb-2">Server Password:</label>
            <div className="flex gap-2">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="flex-grow px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
        <div className="bg-white p-6 rounded-lg shadow-md ">
          
          <h2 className="text-xl font-semibold ">Image Processing</h2>
          
          <div>
            <label className="block text-gray-700 mb-2">Upload Image Password Required</label>

            {/*File Drop Zone */}
            <div
              className="drop-zone flex justify-center items-center border-4 border-dashed border-gray-400 p-8  bg-gray-100 rounded-lg hover:bg-gray-200"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <p className="text-gray-600">Drag & drop your image here, or click to select</p>
            </div>
            {/* Normal File uploader */}

            <div className="flex gap-2 ">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className="flex-grow px-3 py-2 border rounded"
                disabled={isLoading || !passwordVerified}
              />

              {/*Clear Data*/}
              <button
                onClick={clearData}
                className="bg-gray-300 text-gray-700 px-4 py-2 rounded hover:bg-gray-400 disabled:opacity-50"
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

        <div className="bg-white p-6 rounded-lg shadow-md image-section ">
          {/* Uploaded Image */}
          <div>
            <h3 className="text-gray-700 text-lg font-semibold mb-3">Original Image</h3>
            <div className="p-4 bg-gray-50 h-64 flex items-center justify-center overflow-hidden">
              {previewImage ? (
                <img
                  src={previewImage}
                  alt="Original"
                  className="object-contain max-w-full max-h-full image-container"
                />
              ) : (
                <p className="text-gray-500">Upload an image to see preview</p>
              )}
            </div>
          </div>
          </div>
          

          <div className="bg-white p-6 rounded-lg shadow-md image-container ">
          {/* Processed Image */}
          <div>
            <h3 className="text-gray-700 text-lg font-semibold mb-3">Processed Image</h3>
            <div className="p-4 bg-gray-50 h-64 flex items-center justify-center overflow-hidden">
              {processedImage ? (
                <img
                  src={`data:image/jpeg;base64,${processedImage}`}
                  alt="Processed"
                  className="object-contain max-w-full max-h-full image-container"
                />
              ) : (
                <p className="text-gray-500">Processed image will appear here</p>
              )}
            </div>
          </div>

            {/*Highest Emotion Detected */}
            <p className="text-2xl font-bold text-blue-600 mt-2">{detectedEmotion || "[Emotion Here]"}</p>
            <p className="text-2xl font-bold text-blue-600 mt-2">{processStatus || "[Processing Status Here]"}</p>
          </div>
         </div>
      </div>        
      </div>
  );
}

export default App;
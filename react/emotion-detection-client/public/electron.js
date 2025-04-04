const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const waitOn = require('wait-on');

let mainWindow;
let reactProcess = null;

function logMessage(message) {
  console.log(`[ELECTRON] ${message}`);
}

function startReactServer() {
  logMessage('Starting React development server...');
  
  reactProcess = spawn('npx', ['react-scripts', 'start'], {
    shell: true,
    env: {
      ...process.env,
      BROWSER: 'none',
      PORT: 3000
    },
    stdio: 'inherit'
  });

  reactProcess.on('error', (err) => {
    logMessage(`Failed to start React development server: ${err}`);
  });

  reactProcess.on('close', (code) => {
    logMessage(`React development server process exited with code ${code}`);
    reactProcess = null;
  });

  return reactProcess;
}

async function createWindow() {
  logMessage('Creating Electron window...');
  logMessage(`Current directory: ${__dirname}`);
  logMessage(`App path: ${app.getAppPath()}`);
  
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      devTools: false
    },
  });

  // For development
  const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
  logMessage(`Running in ${isDev ? 'development' : 'production'} mode`);

  try {
    if (isDev) {
      logMessage('Waiting for React development server...');
      await waitOn({
        resources: ['http://localhost:3000'],
        timeout: 30000,
      });
      logMessage('React server is ready. Loading application...');
      await mainWindow.loadURL('http://localhost:3000');
      mainWindow.webContents.openDevTools(); // Open DevTools in development mode
    } else {
      logMessage('Loading production build...');
      
      // Try multiple possible paths for the index.html file
      const possiblePaths = [
        path.join(__dirname, 'index.html'),                   // When in the build directory
        path.join(__dirname, '../build/index.html'),          // When running from packaged app
        path.join(app.getAppPath(), 'build/index.html'),      // Using app path
        path.join(process.resourcesPath, 'app/build/index.html'), // In resources
        path.join(process.resourcesPath, 'app.asar/build/index.html') // In asar archive
      ];
      
      let indexPath = null;
      
      for (const testPath of possiblePaths) {
        logMessage(`Checking path: ${testPath}`);
        if (fs.existsSync(testPath)) {
          indexPath = testPath;
          logMessage(`Found index.html at: ${indexPath}`);
          break;
        }
      }
      
      if (!indexPath) {
        logMessage('Could not find index.html in any expected location!');
        app.quit();
        return;
      }
      
      // Load using file protocol
      const fileUrl = `file://${indexPath}`;
      logMessage(`Loading URL: ${fileUrl}`);
      
      await mainWindow.loadURL(fileUrl);
      
      // For debugging in production
      mainWindow.webContents.openDevTools();
    }
  } catch (err) {
    logMessage(`Error during window creation: ${err}`);
    app.quit();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', async () => {
  logMessage('App is ready');
  
  if (process.env.NODE_ENV === 'development' || !app.isPackaged) {
    if (!reactProcess) {
      startReactServer();
    }
  }
  
  await createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('will-quit', () => {
  if (reactProcess) {
    logMessage('Terminating React development server...');
    const isWin = process.platform === 'win32';
    if (isWin) {
      require('tree-kill')(reactProcess.pid);
    } else {
      reactProcess.kill();
    }
  }
});
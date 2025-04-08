import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.example.app',
  appName: 'emotion-detection-client',
  webDir: 'build',
  server: {
    cleartext: true, // Allow cleartext traffic (HTTP)
    // Don't specify the React app URL in `server.url`
  },
};

export default config;

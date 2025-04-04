"Production Mode"

Unzip Production_Apps


In server folder

Run Emotion_Server.exe

Wait for Server Terminal to load

(may take a bit to initialise)

Open port 5000 of the machines ip to access server front end when prompted either by search bar or ctrl clicking links on terminal


In client folder

run Emotion Detection Client.exe

Client App window will open automatically





"Making your own production Apps"


Client

Open react/emotion-detection-client in command line

npm start build

npm start electron-build

go to dist/win_unpacked after completion

run emotion detection client


Server

Run pyinstaller ./emotion_server.spec

go to dist directory after completion

run Emotion_Server

"Dev Mode"

unzip shortened Kaggle emotion dataset AHNSS

use npm install when in react/emotion-detection-client.

this will get all the node modules you need

npm start to start up client on port 3000 of machine


run serverjson.py for the server side

open port 5000 for the html page used for setting a password and starting the server itself


Input private ip into client server field

input server password

Verify Both

Upload your image

Press process and the program does the rest
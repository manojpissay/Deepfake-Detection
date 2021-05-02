from flask_ngrok import run_with_ngrok
from flask import Flask,request,jsonify,abort
from flask_cors import CORS
import os
import json
import requests
from test_image import detect_image
from test_audio import detect_audio
from test_video import detect_video
import urllib.request
import pyrebase
import moviepy.editor as mp

app = Flask(__name__)
run_with_ngrok(app)

CORS(app, resources={r'/*': {'origins': 'http://localhost:8080'}})
CORS(app, resources={r'/*': {'origins': 'https://detect-deepfake.web.app/'}})
CORS(app, resources={r'/*': {'origins': 'https://detect-deepfake.firebaseapp.com/'}})
CORS(app, resources={r'/*': {'origins': '*'}})

firebaseConfig = {
  "apiKey": "AIzaSyAoWFyl24pDPQMupl0SEtvWbPq3VvkK9tE",
  "authDomain": "detect-deepfake.firebaseapp.com",
  "databaseURL": "https://detect-deepfake.firebaseio.com",
  "projectId": "detect-deepfake",
  "storageBucket": "detect-deepfake.appspot.com",
  "messagingSenderId": "599421631503",
  "appId": "1:599421631503:web:798ab942ca267b9eab777a"
};

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

@app.route('/', methods=['GET'])
def server():
    url = "http://localhost:4040/api/tunnels"
    res = requests.get(url)
    res_unicode = res.content.decode("utf-8")
    res_json = json.loads(res_unicode)
    url = res_json["tunnels"][0]["public_url"]
    curr_url = "https://" + url.split('/')[-1]
    print("Server:", curr_url)
    db.child().update({"server_url":curr_url})
    return jsonify('Deepfake Detection Server')

@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/imageDD',methods=['POST'])
def imageDD():
    requestJSON = request.json
    print("Image: ", requestJSON)
    imageName = requestJSON['imageName']
    imageURL = requestJSON['imageURL']
    print(imageName)
    print(imageURL)
    urllib.request.urlretrieve(imageURL, "predict/"+imageName)
    confLevel = detect_image(filename="predict/"+imageName)
    prediction = ""
    if(confLevel>0.5):
        prediction = "Deepfake"
    else:
        prediction = "Pristine" 
    res = {
        "imageURL": imageURL,
        "confLevel": str(confLevel),
        "prediction": prediction
    }
    print("Prediction:",confLevel, sep=" ")
    return res

@app.route('/audioDD',methods=['POST'])
def audioDD():
    requestJSON = request.json
    print("Audio: ", requestJSON)
    audioName = requestJSON['audioName']
    audioURL = requestJSON['audioURL']
    print(audioName)
    print(audioURL)
    urllib.request.urlretrieve(audioURL, "predict/"+audioName)
    confLevel = detect_audio(filename="predict/"+audioName)
    prediction = ""
    if(confLevel == 'No Audio'):
        prediction = "No Audio"
    elif(confLevel>0.5):
        prediction = "Deepfake"
    else:
        prediction = "Pristine"
    res = {
        "audioURL": audioURL,
        "confLevel": str(confLevel),
        "prediction": prediction
    }
    print("Prediction:",confLevel, sep=" ")
    return res

@app.route('/videoDD',methods=['POST'])
def videoDD():
    requestJSON = request.json
    print("Video: ", requestJSON)
    videoName = requestJSON['videoName']
    videoURL = requestJSON['videoURL']
    print(videoName)
    print(videoURL)
    urllib.request.urlretrieve(videoURL, "predict/"+videoName)
    input_clip = mp.VideoFileClip(r"predict/"+videoName) 
    if(input_clip.audio):
        input_clip.audio.write_audiofile(r"predict/"+videoName.split('.')[0]+"_extracted_audio.wav") 
        print("Starting audio deepfake detection...")
        audio_confLevel = detect_audio(filename="predict/"+videoName.split('.')[0]+"_extracted_audio.wav")
        print("Audio deepfake detection completed.")
    else:
        audio_confLevel = 'No Audio'
    print("Starting face and body language deepfake detection...")
    real_face_frames, df_face_frames, real_body_frames, df_body_frames = detect_video("predict/"+videoName)
    res = {
        "videoURL": videoURL,
        "frames": {
            "face": {
                "real_face_frames": str(real_face_frames),
                "df_face_frames": str(df_face_frames)
            },
            "body_lang": {
                "real_body_frames": str(real_body_frames),
                "df_body_frames": str(df_body_frames)
            }
        },
        "audio": {
            "confLevel": str(audio_confLevel)
        }
    }
    print("Deepfake detection results:", res)
    return res

if __name__ == '__main__':
    app.run()
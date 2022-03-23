# Deepfake Detection of Media using Deep Neural Networks

## About
- 3 different neural networks are used to detect any deformity/irregularity in media based on the person's face, audio and body language.
- The face deepfake model uses a Maximum Margin Object Detector (to extract the face) followed by a Temporal Neural Network for classification.
<img width="479" alt="Face" src="https://user-images.githubusercontent.com/45272841/155396030-f56278dc-960f-434e-a030-6d4d8493704d.png">

- Input audio from media is converted into a spectrogram using the librosa library, and then fed to the model which comprises of ResNet50V2 followed by a Temporal Convolutional Network, which predicts whether the given audio is deepfake or not.
- For body language, the entire body of a person is extracted using YOLOv3 followed by a Temporal Neural Network for classification.

## Setup
Install all the dependencies<br>
`pip3 install -r requirements.txt`

## Running the Models


- Sample images, audio and video files are present in the predict folder to test the model
- It is better to run the python programs on a GPU as it may require high computation while running the deep learning models.

Download the project from the GitHub repository<br>
`git clone https://github.com/manojpissay/Deepfake-Detection.git`

A. **Deepfake Detection for Images**:<br>
`python test_image.py -f "path to your file"`

B. **Deepfake Detection for Audio**:<br>
`python test_audio.py -f "path to your file"`

C. **Deepfake Detection for Videos**:<br>
`python test_video.py -f "path to your file"`

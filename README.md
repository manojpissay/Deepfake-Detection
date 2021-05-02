# Deepfake-Detection

Download the project from the GitHub repository by executing the below line.
git clone https://github.com/manojpissay/Deepfake-Detection.git

To predict whether the given input is deepfake or pristine, run the code accordingly for its respective format.

Note:
•	Include the input file to be predicted in the “predict” folder before running the program.
•	Change the name of input file to be predicted in the main function which is there at the bottom of the page in each of the python files.
•	It is better to run the python programs on a GPU as it may require high computation while running the deep learning models.

For Image Deepfake Detection:

•	python test_image.py

For Audio Deepfake Detection:
•	Download the checkpoint from the below link:
    https://drive.google.com/file/d/1vJXh8j3E5TgjMHvegBiFiM9w9V_CsVPs/view?usp=sharing
    Put the download audio checkpoint in "checkpoints" folder.

•	python test_audio.py

For Video Deepfake Detection:

•	python test_video.py

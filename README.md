# Deepfake-Detection

Download the project from the GitHub repository<br>
`git clone https://github.com/manojpissay/Deepfake-Detection.git`

To predict whether the given input is deepfake or pristine, run the code accordingly for its respective format.

- Include the input file to be predicted in the “predict” folder before running the program.
- Change the name of input file to be predicted in the main function which is there at the bottom of the page in each of the python files.
- It is better to run the python programs on a GPU as it may require high computation while running the deep learning models.

A. **Deepfake Detection for Images**:
1. Run `python test_image.py`

B. **Deepfake Detection for Audio**:
1. Download the checkpoint from [here](https://drive.google.com/file/d/1vJXh8j3E5TgjMHvegBiFiM9w9V_CsVPs/view?usp=sharing). Place the downloaded audio checkpoint in "checkpoints" folder.
2. Run `python test_audio.py`

C. **Deepfake Detection for Videos**:
1. Run `python test_video.py`

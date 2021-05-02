from tcn import compiled_tcn, TCN
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import glob
import librosa
import cv2

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Activation, Reshape, Input, BatchNormalization, Flatten
from tensorflow.keras import Sequential
from tensorflow.python.keras.metrics import Metric

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def _linear_to_mel(spectogram):
	  _mel_basis = librosa.filters.mel(16000, 1000, n_mels=240,fmin=0, fmax=8000)
	  return np.dot(_mel_basis, spectogram)

def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def get_melspectrogram(wav):
    D = librosa.stft(y=wav, n_fft=1000, hop_length=200, win_length=800, pad_mode='constant')
    S = _amp_to_db(_linear_to_mel(np.abs(D)**2.)) - 20
    return np.clip((2 * 1.) * ((S - (-100)) / (-(100))) - 1., -(1.), 1.)


def detect_audio(filename = "predict/real_audio1.flac"):
    audio_array,sr = librosa.load(filename, sr=16000)
    trim_audio_array, index = librosa.effects.trim(audio_array)
    S = librosa.feature.melspectrogram(trim_audio_array)
    mels = np.log(S + 1e-9)
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) 
    img = 255-img 
    if img.shape[1]<60:
        print("Audio is too small.")
    itr=0
    x_train = []
    while(itr+60<img.shape[1]):
        img = cv2.cvtColor(img[:,itr:itr+60],cv2.COLOR_GRAY2RGB)
        x_train.append(img[:,itr:itr+60])
        itr+=60
    x_train = np.array(x_train)
    print(x_train.shape)
    x_train = x_train.astype('float32')
    model = Sequential()
    model.add(Input(shape=(128,60,3)))
    model.add(ResNet50V2(include_top=False))
    model.add(Reshape(( 16384, 1)))
    model.add(TCN(nb_filters=64,
        kernel_size=3,
        dropout_rate=0.1,
        dilations=[1, 2, 4, 8, 16, 32],
        use_batch_norm=True,
        return_sequences=False)
      )
    model.add(Dense(16))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005), loss='binary_crossentropy',
      metrics=[ 'acc',tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall() ])
    model.summary()    
    model.load_weights('checkpoints/audio_checkpoint.hdf5')
    pred = model.predict(x_train)
    prediction = 1 - np.average(pred)
    print("Model prediction for:",filename,sep=" ")
    print("Confidence of it being a deepfake:",prediction)
    return prediction


if __name__ == '__main__':
    detect_audio(filename = "predict/df_audio.flac")

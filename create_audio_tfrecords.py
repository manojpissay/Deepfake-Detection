import os
import numpy as np
import librosa
import cv2
import tensorflow as tf

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(audio, label, width, height, channels):
    feature = {
        # 'audio': _float_feature(audio),  # audio is a list of floats
        'audio': _floats_feature(audio),
        'label': _int_feature([label]),  # wrap label index in list
        'width': _int_feature([width]),
        'height': _int_feature([height]),
        'channels': _int_feature([channels])

    }
    # Example is a flexible message type that contains key-value pairs,
    # where each key maps to a Feature message. Here, each Example contains
    # two features: A FloatList for the decoded audio data and an Int64List
    # containing the corresponding label's index.
    return tf.train.Example(features=tf.train.Features(feature=feature))

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

if __name__ == "__main__":
    
    audio_dir = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA_Full_Dataset/LA/ASVspoof2019_LA_dev/"
    
    f = open("/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA_Full_Dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")

    
    lines = f.readlines()
    
    lines=lines[2548:]
    lines=lines[3*len(lines)//4:]
    c=0
    with tf.io.TFRecordWriter('/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/audio_tfrecords/dev_spoof_4.tfrecord') as out:
        
        for count,line in enumerate(lines):
            filename = line.split()[1].strip()
            label = line.split()[-1].strip()
            if label!="spoof":
                continue
            label = 0

            audio_data = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA_Full_Dataset/LA/ASVspoof2019_LA_dev/flac/" + filename + ".flac" 
            audio_array,sr = librosa.load(audio_data, sr=16000)
            trim_audio_array, index = librosa.effects.trim(audio_array)
            S = librosa.feature.melspectrogram(trim_audio_array)
            mels = np.log(S + 1e-9) # add small number to avoid log(0)
            # min-max scale to fit inside 8-bit range
            img = scale_minmax(mels, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0) # put low frequencies at the bottom in image
            img = 255-img # invert. make black==more energy
            if img.shape[1]<60:
                continue
            itr=0
            c+=1
            print(c, filename)
            
            # if img.shape[1]==60:
                # print("shape===60: ",audio_data)

            while(itr+60<img.shape[1]):
                img = cv2.cvtColor(img[:,itr:itr+60],cv2.COLOR_GRAY2RGB)
                # img = tf.io.encode_jpeg(img)
                # img = tf.io.serialize_tensor(img)
                tfexample = to_tfrecord(img, label,img.shape[0],img.shape[1],img.shape[2])
                out.write(tfexample.SerializeToString())
                itr+=60

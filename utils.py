import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from PIL import Image
import io
import json
import os
import IPython.display as display
import cv2
import librosa
import librosa.display
import math
import random
import re

# DIR = "/content/drive/My Drive/Deepfake Detection/data/deepfake_images/DeepFake00/DeepFake00"
DIR = "/content/drive/My Drive/Deepfake Detection/data/deepfake_images/"

def read_audio_tfrecords():
    tfrecords = []
    data_dir = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/audio_tfrecords/"
    for tfrecord in sorted(os.listdir(data_dir)):
        if tfrecord.endswith('.tfrecord') and tfrecord[:3]=='dev':
            print(tfrecord)
            tfrecords.append(os.path.join(data_dir, tfrecord))
    real_raw_audio_dataset = tf.data.TFRecordDataset(tfrecords)
    audio_feature_description = {
            # 'audio': tf.io.FixedLenFeature([], tf.float32)
            'audio': tf.io.FixedLenSequenceFeature([],tf.float32,allow_missing =True),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64)
        }
        
    def _parse_audio_function(serialized):
          parsed_features = tf.io.parse_single_example(serialized, audio_feature_description)
          img = tf.reshape(parsed_features['audio'],[parsed_features['width'],parsed_features['height'],parsed_features['channels']])
          # audio = tf.io.decode_raw(parsed_features['audio'],out_type=float)
          # audio = tf.cast(parsed_features['audio'], tf.float32)
          label = tf.reshape(parsed_features['label'],[1,1])
          return img,parsed_features['label']

    dataset = real_raw_audio_dataset.map(_parse_audio_function)
    fake = dataset.filter(lambda x,y: y == 0)
    real = dataset.filter(lambda x,y: y == 1)
    
    # dev- total: 21309    bonafide-2442    spoof-18867
    real = real.repeat(5)
    real = real.take(10000)
    fake = fake.take(10000)    
    ##To interleave real and fake
    dataset = tf.data.Dataset.zip((real, fake)).flat_map(
        lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
        tf.data.Dataset.from_tensors(x1)))

    return dataset

def label_img(img,subdir):
    digit = int(subdir[-2:])
    f = open("/content/drive/My Drive/Deepfake Detection/data/metadata/metadata"+str(digit)+".json", "r") 
    data = json.loads(f.read().strip())
    v = img.replace(".jpg",".mp4")
    return data[v]["label"],data[v]["split"]

def loadData():
    print("In loadData")
    train_data = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    count = 0
    for subdir, dirs, files in os.walk(DIR):
        for img in files:
            print(count,": ",img)
            label,split = label_img(img,subdir)
            if(label=="REAL"):
                label = 1
            elif(label=="FAKE"):
                label = 0
            path = os.path.join(subdir, img)
            img = Image.open(path)
            img = img.convert('L')
            img = np.array(img)
            img = img.astype("uint8")
            if(split=="test"):
                x_test.append(img)
                y_test.append(label)
            else:
                x_train.append(img)
                y_train.append(label)
            count += 1

            if count==2:
                break
        if count==2:
            break
            # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            # train_data.append([np.array(img), label])
            # Basic Data Augmentation - Horizontal Flipping
            # flip_img = Image.open(path)
            # flip_img = flip_img.convert('L')
            # flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            # flip_img = np.array(flip_img)
            # flip_img = np.fliplr(flip_img)
            # train_data.append([flip_img, label])
        # shuffle(train_data)
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

def generate_batches(files, batch_size):
   counter = 0
   while True:
     fname = files[counter]
     print(fname)
     counter = (counter + 1) % len(files)
     data_bundle = pickle.load(open(fname, "rb"))
     X_train = data_bundle[0].astype(np.float32)
     y_train = data_bundle[1].astype(np.float32)
     y_train = y_train.flatten()
     for cbatch in range(0, X_train.shape[0], batch_size):
         yield (X_train[cbatch:(cbatch + batch_size),:,:], y_train[cbatch:(cbatch + batch_size)])

def getTrainData():
    tfrecords = []
    data_dir = "/content/drive/My Drive/Deepfake Detection/data/tfrecords"
    count = 0
    for tfrecord in sorted(os.listdir(data_dir)):
        if tfrecord.endswith('.tfrecords'):
            count += 1
            if(count>25):
                break
            print(tfrecord)
            tfrecords.append(os.path.join(data_dir, tfrecord))


    # raw_image_dataset = tf.data.TFRecordDataset("/content/drive/My Drive/Deepfake Detection/data/tfrecords/DeepFake0.tfrecords")
    raw_image_dataset = tf.data.TFRecordDataset(tfrecords)

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'split': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    # image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # depth = tf.cast(features['depth'], tf.int32)
    # label = tf.cast(features['label'], tf.int32)
    # split = tf.cast(features['split'], tf.int32)

    def _parse_image_function(example_proto):
      return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    print("Parsed Image Dataset:",parsed_image_dataset)
    # c=0

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    # parsed_image_dataset = parsed_image_dataset.batch(128).prefetch(10).take(5)
    # l = len(parsed_image_dataset)

    l = 0
    r = 0
    f =0
    real =0
    fake =0
    for image_features in parsed_image_dataset:
        l += 1
        label = image_features['label']
        if label == 1:
          r+=1
        else:
          f+=1
    print("Length:",l)
    dc = 0
    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        # print(image_raw.shape)
        # print(type(image_raw))
        img = Image.open(io.BytesIO(image_raw))
        image = np.asarray(img)
        # image = tf.cast(tf.io.decode_raw(image_raw, tf.uint8), tf.float32).numpy()
        # image = np.fromstring(image_raw, dtype=np.uint8)
        # print("Shape before reshaping:",image.shape,image)
        #We have to convert it into (270, 480,3) in order to see as an image
        # image = image.reshape((150,150,3))
        img = np.dot(image[...,:3], [0.299, 0.587, 0.114]) # Converting RGB to Grayscale
        # print("img Shape:",image.shape)
        label = image_features['label']
        if(label==1):
          real+=1
        else:
          fake+=1
        if(fake>=r and label==0):
          continue
        if(dc<0.8*(r*2)): # If split is test
            x_train.append(img)
            y_train.append(label)
        else:
            x_test.append(img)
            y_test.append(label)
        dc += 1
        # print(c)
        # f = open("/content/drive/My Drive/img/img{}.jpg".format(c),'wb')
        # c+=1
        # f.write(image_raw)
        # cv2.imwrite("images/img1.jpg",image_raw)
        # display.display(display.Image(data=image_raw))
        # break
    print("fake and real ",fake,real)
    print("xtrain and ytrain", len(x_train), len(y_train))
    print("xtest and ytest", len(x_test), len(y_test))
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
    

def getTrainDataBatch():
    tfrecords = []
    data_dir = "/content/drive/My Drive/Deepfake Detection/data/tfrecords"
    count = 0
    # for tfrecord in sorted(os.listdir(data_dir)):
    #     count += 1
    #     if tfrecord.endswith('.tfrecords'):
    #         if(count<25):
    #             continue
    #         print(tfrecord)
    #         tfrecords.append(os.path.join(data_dir, tfrecord))
    for tfrecord in sorted(os.listdir(data_dir)):
        count += 1
        if tfrecord.endswith('.tfrecords'):
            if(count>=40):
                break
            print(tfrecord)
            tfrecords.append(os.path.join(data_dir, tfrecord))


    # raw_image_dataset = tf.data.TFRecordDataset("/content/drive/My Drive/Deepfake Detection/data/tfrecords/DeepFake0.tfrecords")
    
    raw_image_dataset = tf.data.TFRecordDataset(tfrecords)

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'split': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    def _parse_image_function(serialized):
      parsed_features = tf.io.parse_single_example(serialized, image_feature_description)
      # print("TYPE IMAGE_RAW: ",type(parsed_features['image_raw']))
      # print("IMAGE_RAW: ",parsed_features['image_raw'])
      # parsed_features_train = parsed_features['image_raw'].numpy()
      img = tf.image.decode_jpeg(parsed_features['image_raw'], channels=3)
      img = tf.image.resize(img, [150, 150])
      
      img = tf.tensordot(img, tf.constant([0.299, 0.587, 0.114]), 1)
      img = tf.reshape(img,[150*150,1])
      # img = tf.reshape(img,[150,150,3])
      
      # img = tf.keras.applications.efficientnet.preprocess_input(img)
      # img = tf.reshape(img,[1,150*150,3])
      
      # print("Decoded: ",img)
      # parsed_features_train = tf.io.decode_raw(parsed_features['image_raw'], tf.uint8)
      # parsed_features_train = np.dot(parsed_features_train[...,:3], [0.299, 0.587, 0.114]) # Converting RGB to Grayscale
      #print("ffew",parsed_features_train)
      # print("Converted IMAGE-----------------: ",parsed_features_train,tf.shape(parsed_features_train) )
      # parsed_features_train = tf.reshape(parsed_features_train[0], [150 * 150, 1])
      #parsed_features_train = tf.reshape(parsed_features['image_raw'], [150 * 150, 1])
      # parsed_features_train = parsed_features['image_raw'].reshape(150 * 150, 1)
      # img = preprocess_input(img)
      parsed_features_train = img
      # num_classes = 2
      # tf.print("Y",parsed_features['label'])
      #print(tf.compat.v1.Session().run(parsed_features['label']))
      # y_train = to_categorical([parsed_features['label']], num_classes)
      # y_train = np.expand_dims(y_train, axis=2)

      # parsed_features_train = parsed_features_train.astype('float32')
      parsed_features_train /= 255
      # l2=tf.cast(tf.constant(1),dtype=tf.int64)
      # label = tf.stack([parsed_features['label'], l2],axis=-1)
      # label =  tf.reshape(label,[1,2])
      # label = tf.reshape(parsed_features['label'],[1,1])
      # return label,parsed_features_train
      return parsed_features_train, parsed_features['label']

    dataset = raw_image_dataset.map(_parse_image_function)
    # print("Parsed Image Dataset:",dataset)

    ##Fix imbalanced data
    fake = dataset.filter(lambda x,y: y == 0)
    real = dataset.filter(lambda x,y: y == 1)
    real=real.repeat(4)
    fake=fake.take(50000)
    real=real.take(50000)

    ##To interleave real and fake
    dataset = tf.data.Dataset.zip((real, fake)).flat_map(
        lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
        tf.data.Dataset.from_tensors(x1)))
    
    return dataset


def data_generator():
    # input image dimensions
    img_rows, img_cols = 150, 150
    # img_rows, img_cols = 28, 28
    # (x_train, y_train), (x_test, y_test) = loadData()
    (x_train, y_train), (x_test, y_test) = getTrainData()
    # (x_train, y_train), (x_test, y_test) = getTrainDataBatch()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print("Shape:",x_train.shape)
    x_train = x_train.reshape(-1, img_rows * img_cols, 1)
    x_test = x_test.reshape(-1, img_rows * img_cols, 1)

    num_classes = 2
    # num_classes = 10
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

def audio_data_generator():
    audio_dir = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_train/"
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    f = open("/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")

    b = 0
    s = 0
    bonafide = 0
    spoof = 0
    dc = 0

    lines = f.readlines()
    # l = lines[:501] + lines[2581:3081]
    # l = lines[:101] + lines[2981:3081]
    l = lines[:2580] + lines[2600:5180]
    for line in l:
        filename = line.split()[1].strip()
        print(dc, filename)
        label = line.split()[-1].strip()

        audio_data = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_train/flac/" + filename + ".flac" 
        y,sr = librosa.load(audio_data, sr=None)

        # max_length = 80000
        # max_length = 150000
        # if(y.shape[0]>max_length):
        #     y = y[:max_length]
        # else:
        #     silence = np.zeros(max_length-y.shape[0],)
        #     y = np.concatenate((y,silence))
        # print(y.shape)

        n_mels = 64
        n_fft = int(np.ceil(0.025*sr))
        win_length = int(np.ceil(0.025*sr))
        hop_length = int(np.ceil(0.010*sr))
        window = 'hamming'
        fmin = 20
        fmax = 8000
        S = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
        frames = np.log(librosa.feature.mfcc(y=y, sr=sr, S=S, n_mels=n_mels, fmin=fmin, fmax=fmax) + 1e-6)
        
        # print("Affter mfcc",frames.shape)
        # dimension_frame=frames.shape[1]
        # pad_dimension=(100-dimension_frame%100)
        # silence = np.zeros((20,pad_dimension))
        # frames=np.concatenate((frames,silence),axis=1)
        # print("Affter pad",frames.shape)
        
        # window_size = 64
        # window_hop = 30

        # truncate at start and end to only have windows full data
        # alternative would be to zero-pad
        # start_frame = window_size 
        # end_frame = window_hop * math.floor(float(frames.shape[1]) / window_hop)

        if(label=="bonafide"):
            label = 1
            b += 1
        else:  
            label = 0
            s += 1

        if(label==1):
            bonafide += 1
        else:
            spoof += 1
        print(b,s,sep=" ")

        # if(dc<=800):
        if(dc<4128):
            x_train.append(frames)
            y_train.append(label)
        else:
            x_test.append(frames)
            y_test.append(label)

        # ctr = 0
        # for frame_idx in range(start_frame, end_frame, window_hop):
        #     window = frames[:, frame_idx-window_size:frame_idx]
        #     if(dc<=400):
        #         x_train.append(window)
        #         y_train.append(label)
        #     else:
        #         x_test.append(window)
        #         y_test.append(label)
        #     ctr+=1
        #     # print('classify window', frame_idx, window.shape)
        dc += 1

    ##Padding
    max_dimension=0
    for i in range(len(x_train)):
        element_shape=x_train[i].shape[1]
        max_dimension=max(max_dimension,element_shape)
    print("Max length",max_dimension)
    
    for i in range(len(x_train)):
        # print("Before pad",x_train[i].shape)
        cur_dimension=x_train[i].shape[1]
        # pad_dimension=(100-dimension_frame%100)
        silence = np.zeros((20,max_dimension-cur_dimension))
        x_train[i]=np.concatenate((x_train[i],silence),axis=1)
    print("After pad",x_train[i].shape)
    
    max_dimension_test=0
    for i in range(len(x_test)):
        element_shape=x_test[i].shape[1]
        max_dimension_test=max(max_dimension_test,element_shape)
    # print("Max length",max_dimension)
    
    for i in range(len(x_test)):
        # print("Before pad",x_train[i].shape)
        cur_dimension=x_test[i].shape[1]
        # pad_dimension=(100-dimension_frame%100)
        silence = np.zeros((20,max_dimension_test-cur_dimension))
        x_test[i]=np.concatenate((x_test[i],silence),axis=1)
        # print("Affter pad",x_train[i].shape)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # print("inside",x_train.shape)
    audio_rows = 20
    x_train = x_train.reshape(-1, audio_rows * max_dimension, 1)
    x_test = x_test.reshape(-1, audio_rows * max_dimension_test, 1)
    
    
    # print("Shape of x_train:",x_train.shape)
    # print("Shape of y_train:",y_train.shape)
    # x_train = x_train.reshape(len(x_train),20,max_dimension)
    # x_test = x_test.reshape(len(x_test),20,max_dimension_test)
    # x_train=x_train.reshape((x_train.shape[0],20*max_dimension,1))
    # x_test=x_test.reshape((x_test.shape[0],20*max_dimension_test,1))
    
    # x_test = x_test.reshape((len(x_test), max_dimension_test, 1))

    num_classes = 2
    # num_classes = 10
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # y_train = np.reshape(y_train, (-1, 1))
    # y_test = np.reshape(y_test, (-1, 1))


    # y_train = np.expand_dims(y_train, axis=2)
    # print("y_train:",y_train)
    # y_test = np.expand_dims(y_test, axis=2)

    y_train = np.expand_dims(y_train, axis=1)
    # print("y_train:",y_train)
    y_test = np.expand_dims(y_test, axis=1)

    # y_train=y_train.reshape((len(y_train)1,2,1))
    # y_test=y_test.reshape((len(y_test),2,1))
    
    # y_train = x_train.reshape(20,max_dimension,len(x_train))
    # y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    # y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')


    return (x_train, y_train), (x_test, y_test)


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

def get_spectogram():
    audio_dir = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_train/"
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    f = open("/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")

    b = 0
    s = 0
    bonafide = 0
    spoof = 0
    dc = 0

    lines = f.readlines()
    l = lines[:2580] + lines[2600:5180]
    # l = lines[:101] + lines[2981:3081]
    # l = lines[:6] + lines[3075:3081]
    random.shuffle(l)
    for line in l:
        filename = line.split()[1].strip()
        print(dc, filename)
        label = line.split()[-1].strip()

        audio_data = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_train/flac/" + filename + ".flac" 
        audio_array,sr = librosa.load(audio_data, sr=16000)
        trim_audio_array, index = librosa.effects.trim(audio_array)
        S = get_melspectrogram(trim_audio_array).T
        # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

        # print(type(S))
        # print(S.shape)
        max_dimension = 240
        max_dimension_test = 240
        curr_dimension = S.shape[0]
        if(max_dimension_test>curr_dimension):
            silence = np.zeros((max_dimension_test-curr_dimension,240))
            S = np.concatenate((S,silence),axis=0)
        else:
            S = S[:240,:]
        print(S.shape)

        if(label=="bonafide"):
            label = 1
        else:  
            label = 0

        if(dc<=4128):
            x_train.append(S)
            y_train.append(label)
        else:
            x_test.append(S)
            y_test.append(label)
        dc += 1
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train.reshape(-1, 240 * max_dimension, 1)
    x_test = x_test.reshape(-1, 240 * max_dimension_test, 1)

    num_classes = 2

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def get_spectogram_new():
    audio_dir = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_train/"
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    f = open("/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")

    b = 0
    s = 0
    bonafide = 0
    spoof = 0
    dc = 0

    lines = f.readlines()
    l = lines[:2580] + lines[2600:5180]
    # l = lines[:101] + lines[2981:3081]
    # l = lines[:6] + lines[3075:3081]
    random.shuffle(l)
    for line in l:
        filename = line.split()[1].strip()
        print(dc, filename)
        label = line.split()[-1].strip()

        audio_data = "/content/drive/My Drive/Deepfake Detection/data/deepfake_audio/LA/ASVspoof2019_LA_train/flac/" + filename + ".flac" 
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

        if(label=="bonafide"):
            label = 1
        else:  
            label = 0
        
        while(itr+60<img.shape[1]):
            img = cv2.cvtColor(img[:,itr:itr+60],cv2.COLOR_GRAY2RGB)
            x_train.append(img[:,itr:itr+60])
            y_train.append(label)
            itr+=60
        
        dc += 1
    
    x_test= x_train[:int(0.2*len(x_train))]
    x_train= x_train[int(0.2*len(x_train)):]

    y_test = y_train[:int(0.2*len(y_train))]
    y_train = y_train[int(0.2*len(y_train)):]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # x_train = x_train.reshape(-1, 128 * 60, 1)
    # x_test = x_test.reshape(-1, 128 * 60, 1)

    # num_classes = 2

    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)
    # y_train = np.expand_dims(y_train, axis=1)
    # y_test = np.expand_dims(y_test, axis=1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)

'''
def body_data_generator():
    real_body_dir = "/content/drive/My Drive/Deepfake Detection/data/body_language/Final Dataset/Real"
    fake_body_dir = "/content/drive/My Drive/Deepfake Detection/data/body_language/Final Dataset/Fake"
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    for vid in os.listdir(real_body_dir):
        vidcap = cv2.VideoCapture(real_body_dir+"/"+vid)
        cv2.waitKey(delay_time)
        success,image = vidcap.read()
        count = 0
        while success:
            count += 1
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            if(count%60!=0):
                continue
            x_train.append(img)
            y_train.append(1)

    x_test = x_train[int(len(x_train)*0.9):]
    y_test = y_train[int(len(y_train)*0.9):]

    x_train = x_train[:int(len(x_train)*0.9)]
    y_train = y_train[:int(len(y_train)*0.9)]

    print("Real body dir: x_train",len(x_train))

    for vid in os.listdir(fake_body_dir):
        vidcap = cv2.VideoCapture(fake_body_dir+"/"+vid)
        success,image = vidcap.read()
        count = 0
        while success:
            count += 1
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            if(count%60!=0):
                continue
            x_train.append(img)
            y_train.append(0)

    x_test += x_train[int(len(x_train)*0.9):]
    y_test = y_train[int(len(y_train)*0.9):]

    x_train = x_train[:int(len(x_train)*0.9)]
    y_train = y_train[:int(len(y_train)*0.9)]

    print("Fake body dir: x_train",len(x_train))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    img_rows = 150
    img_cols = 150

    x_train = x_train.reshape(-1, img_rows * img_cols, 1)
    x_test = x_test.reshape(-1, img_rows * img_cols, 1)

    num_classes = 2
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(x_test.shape[0])
    np.random.shuffle(indices)

    x_test = x_test[indices]
    y_test = y_test[indices]

    return (x_train, y_train), (x_test, y_test)
'''
def get_Body_Data():
    real_tfrecords = []
    real_datasets=[]
    fake_datasets=[]
    real_data_dir = "/content/drive/My Drive/Deepfake Detection/data/body_language/TFRecords/Real"
    for tfrecord in sorted(os.listdir(real_data_dir)):
        if tfrecord.endswith('.tfrecords'):
            print(tfrecord)
            d=tf.data.TFRecordDataset(os.path.join(real_data_dir, tfrecord))
            d=d.take(4000)
            real_datasets.append(d)
            real_tfrecords.append(os.path.join(real_data_dir, tfrecord))

    fake_tfrecords = []
    fake_data_dir = "/content/drive/My Drive/Deepfake Detection/data/body_language/TFRecords/Fake"
    for tfrecord in sorted(os.listdir(fake_data_dir)):
        if tfrecord.endswith('.tfrecords'):
            print(tfrecord)
            d=tf.data.TFRecordDataset(os.path.join(fake_data_dir, tfrecord))
            d=d.take(1300)
            fake_datasets.append(d)
            fake_tfrecords.append(os.path.join(fake_data_dir, tfrecord))

    real_raw_image_dataset = real_datasets[0]
    for i in real_datasets[1:]:
        real_raw_image_dataset=real_raw_image_dataset.concatenate(i)
    # fake_raw_image_dataset = fake_datasets[0].concatenate(i for i in fake_datasets[1:])  
    fake_raw_image_dataset = fake_datasets[0]
    for i in fake_datasets[1:]:
        fake_raw_image_dataset=fake_raw_image_dataset.concatenate(i)
    # real_raw_image_dataset = tf.data.TFRecordDataset(real_tfrecords)
    # fake_raw_image_dataset = tf.data.TFRecordDataset(fake_tfrecords)

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }


    def _parse_image_function_real(serialized):
      parsed_features = tf.io.parse_single_example(serialized, image_feature_description)
      img = tf.image.decode_jpeg(parsed_features['image_raw'], channels=3)
      # Original IMage Dimensions
      # img = tf.image.resize(img, [512, 1024])
      img = tf.image.resize(img, [150,150])
      img = tf.tensordot(img, tf.constant([0.299, 0.587, 0.114]), 1)
      img = tf.reshape(img,[150*150,1])
      parsed_features_train = img
      num_classes = 2
      parsed_features_train /= 255
      # label = tf.reshape(parsed_features['label'],[1,1])
      # label = tf.convert_to_tensor([1], dtype=tf.int64)
      # label = tf.reshape(label,[2,1])
      return parsed_features_train, 1

    def _parse_image_function_fake(serialized):
      parsed_features = tf.io.parse_single_example(serialized, image_feature_description)
      img = tf.image.decode_jpeg(parsed_features['image_raw'], channels=3)
      img = tf.image.resize(img, [150, 150])
      img = tf.tensordot(img, tf.constant([0.299, 0.587, 0.114]), 1)
      img = tf.reshape(img,[150*150,1])
      parsed_features_train = img
      num_classes = 2
      parsed_features_train /= 255
      # label = tf.reshape(parsed_features['label'],[1,1])
      #Hard Coded
      label = tf.convert_to_tensor([0], dtype=tf.int64)
      # label = tf.reshape(label,[2,1])
      return parsed_features_train, 0

    real_dataset = real_raw_image_dataset.map(_parse_image_function_real)
    fake_dataset = fake_raw_image_dataset.map(_parse_image_function_fake)

    # real_dataset = real_dataset.shuffle(250, reshuffle_each_iteration=True)
    # fake_dataset = fake_dataset.shuffle(250, reshuffle_each_iteration=True)
    
    # real_dataset = real_dataset.take(30000)
    # fake_dataset = fake_dataset.take(30000)

    dataset = tf.data.Dataset.zip((real_dataset, fake_dataset)).flat_map(
        lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
        tf.data.Dataset.from_tensors(x1)))
    
    # dataset = dataset.shuffle(buffer_size=256)
    # dataset = dataset.take(50)
    # print(dataset)
    return dataset


def body_data_generator():
    
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    real_body_dir = "/content/drive/My Drive/Deepfake Detection/data/body_language/List Dataset/Real"
    fake_body_dir = "/content/drive/My Drive/Deepfake Detection/data/body_language/List Dataset/Fake"
    
    for img_name in os.listdir(real_body_dir):
        #read in grayscale
        image = cv2.imread(os.path.join(real_body_dir,img_name), 0) 
        
        ##Resize to 256*256 since fake images are of size 256*256
        # image = cv2.resize(image, (256, 256))
        image = cv2.resize(image, (150, 150))


        x_train.append(image)
        # real -> 1
        y_train.append(1)

    for img_name in os.listdir(fake_body_dir):
        #read in grayscale
        # image = cv2.imread(os.path.join(fake_body_dir,img_name), 0)
        image = cv2.imread(os.path.join(fake_body_dir,img_name), 0) 
        
        ##Resize to 256*256 since fake images are of size 256*256
        # image = cv2.resize(image, (256, 256))
        image = cv2.resize(image, (150, 150))
        x_train.append(image)
        # real -> 1
        y_train.append(0)

    temp = list(zip(x_train, y_train))
    random.shuffle(temp)
    x_train, y_train = zip(*temp)
    
    #Reduce images for temporary arrangements
    # x_train= x_train[int(len(x_train)*0.5):]
    # y_train= y_train[int(len(y_train)*0.5):]

    x_test = x_train[int(len(x_train)*0.8):]
    x_train = x_train[:int(len(x_train)*0.8)]

    y_test = y_train[int(len(y_train)*0.8):]
    y_train = y_train[:int(len(y_train)*0.8)]

    print("Shape:",x_train[0].shape)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Afet Shape:",x_train.shape)

    img_rows = 150
    img_cols = 150
    x_train = x_train.reshape(-1, img_rows * img_cols, 1)
    x_test = x_test.reshape(-1, img_rows * img_cols, 1)

    num_classes = 2
    # num_classes = 10
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)



if __name__ == '__main__':
    # print(data_generator())
    getTrainDataBatch()
    # (x_train, y_train), (x_test, y_test) = body_data_generator()
    # print(x_train.shape,y_train.shape)
    # print(x_test.shape,y_test.shape)
    # audio_data_generator()
    # (x_train, y_train), (x_test, y_test) = audio_data_generator()
    # (x_train, y_train), (x_test, y_test) = get_spectogram()
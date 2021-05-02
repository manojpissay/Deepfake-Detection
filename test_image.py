# from utils import data_generator

from tcn import compiled_tcn
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import glob
import os
import json

from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense,Activation, Reshape, Input, BatchNormalization
from keras.models import Model
from tensorflow.keras.models import Sequential

def detect_image(filename = "predict/test_image.jpg"):
    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=None,
                         use_skip_connections=True,
                         use_batch_norm = True)

    model.summary()
    model.load_weights('checkpoints/image_checkpoint.hdf5')
    test_img = Image.open(filename)
    test_img = test_img.convert("L")
    newsize = (150,150) 
    test_img = test_img.resize(newsize) 
    test_img = np.array(test_img)
    test_images = []  
    test_images.append(test_img)
    test_images = np.array(test_images)
    img_rows, img_cols = 150, 150
    test_images = test_images.reshape(-1, img_rows * img_cols, 1)
    test_images = test_images.astype('float32')
    test_images /= 255
    p = model.predict(test_images)
    print(p)
    prediction = p[0][0]
    print("Model prediction for:",filename,sep=" ")
    print("Confidence of it being a deepfake:",prediction)
    return prediction


def test_tfrecords(tfrecord_number):
    model1 = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=None,
                         use_skip_connections=True,
                         use_batch_norm = True)

    model1.summary()
    model1.load_weights('checkpoints/image_checkpoint.hdf5')

    correct=0
    wrong = 0
    not_there = 0
    f = open("/content/drive/My Drive/Deepfake Detection/data/metadata/metadata"+str(tfrecord_number)+".json")
    img_dict = json.load(f)

    x=[]
    y=[]
    real_images=[]
    fake_images=[]
    count=0
    real=0
    fake=0
    tp=0
    fp=0
    tn=0
    fn=0


    for filename,val in img_dict.items():
        count+=1
        if count>1500:
            break
        x_eval=[]
        filename = filename.replace("mp4","jpg")
        print(filename)
        label = val["label"]
        if(label=="REAL"):
            label = 1
        elif(label=="FAKE"):
            label = 0

        # y.append(label)
        
        img_data = "/content/drive/My Drive/Deepfake Detection/data/deepfake_images/DeepFake"+str(tfrecord_number).zfill(2)+"/DeepFake"+str(tfrecord_number).zfill(2)+"/" + filename
        try:
            test_img = Image.open(img_data)
            if label == 1:
              real+=1
            else:
              fake+=1
        except:
            print("Not found!")
            not_there+=1
            continue

        test_img = test_img.convert("L")
        newsize = (150,150) 
        test_img = test_img.resize(newsize) 
        test_img = np.array(test_img)
        if label ==1:
            real_images.append(test_img)
        else:
            fake_images.append(test_img)

    fake_images = fake_images[:len(real_images)]
    
    real_images = np.array(real_images)
    img_rows, img_cols = 150, 150
    real_images = real_images.reshape(-1, img_rows * img_cols, 1)
    real_images = real_images.astype('float32')
    real_images /= 255

    fake_images = np.array(fake_images)
    img_rows, img_cols = 150, 150
    fake_images = fake_images.reshape(-1, img_rows * img_cols, 1)
    fake_images = fake_images.astype('float32')
    fake_images /= 255

    real_prediction = model1.predict(real_images)
    for i in real_prediction:
        
        pred=0
        if i[0]>0.5:
            pred=1##deepfake

        #confusion matrix
        if pred==0:
            tn+=1
        else:
            fp+=1

    fake_prediction = model1.predict(fake_images)
    for i in fake_prediction:
        pred=0
        if i[0]>0.5:
            pred=1##deepfake

        #confusion matrix
        if pred==0:
            fn+=1
        else:
            tp+=1
    
    ##Inverted due to real ==1 and fake =0
    print("True Positive:",tn)
    print("False Positive:",fn)
    print("True Negative:",tp)
    print("False Negative:",fp)
    print("Real: ",real,"Fake: ",real)

if __name__ == '__main__':
    detect_image(filename = "predict/df_image.jpg")
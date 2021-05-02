import tensorflow as tf
import os
import glob
import numpy as np
import cv2
import json
import sys

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def label_img(img,digit):
    f = open("/content/drive/My Drive/Deepfake Detection/data/metadata/metadata"+str(digit)+".json", "r") 
    data = json.loads(f.read().strip())
    v = img.replace(".jpg",".mp4")
    return data[v]["label"],data[v]["split"]

def image_example(image_string, label, split):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'split': _int64_feature(split),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def printProgressBar(i,max,postText):
    n_bar = 20
    j = i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

for i in range(48,50):
    print("Started creating:",i)
    idx = str(i)
    record_file = "/content/drive/My Drive/Deepfake Detection/data/tfrecords/DeepFake"+idx+".tfrecords"
    with tf.io.TFRecordWriter(record_file) as writer:
        if(i<10):
          idx = "0" + idx
        DIR = "/content/drive/My Drive/Deepfake Detection/data/deepfake_images/DeepFake"+idx+"/DeepFake"+idx+"/"
        files = glob.glob(DIR+"*.jpg")
        c = 0
        t = len(files)
        for filename in files:
            printProgressBar(c,t,"Completed")
            image_string = open(filename, 'rb').read()
            filename = filename.split("/")[-1]
            label,split = label_img(filename,i)
            label_v = 0
            split_v = 0
            if(label=="REAL"):
                label_v = 1
            else:
                label_v = 0
            if(split=="train"):
                split_v = 0
            else:
                split_v = 1
            tf_example = image_example(image_string, label_v, split_v)
            writer.write(tf_example.SerializeToString())
            c+=1
    print("Completed creating:",i)
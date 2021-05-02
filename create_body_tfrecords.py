import tensorflow as tf
import os
import glob
import numpy as np
import cv2
import json
import sys
import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def printProgressBar(i,max,postText):
    n_bar = 20
    j = i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

label = 1
record_file = "/content/drive/My Drive/Deepfake Detection/data/body_language/TFRecords/3.tfrecords"
with tf.io.TFRecordWriter(record_file) as writer:

    DIR = "/content/drive/My Drive/Deepfake Detection/data/body_language/subject4/subject4/train/train_img/"
    files = glob.glob(DIR+"*.png")
    c=0
    t=len(files)
    print("Num of images",len(files))
    for filename in files:
        printProgressBar(c,t,"Completed")
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())
        c+=1

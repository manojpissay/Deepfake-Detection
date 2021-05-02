from tcn import compiled_tcn
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import glob
import face_recognition
import os
import cv2
from test_body import detect_video_body
import shutil

def detect_video_face(img, face_model):
    test_img = img.convert("L")
    test_img = test_img.resize((150,150)) 
    test_img = np.array(test_img)
    test_images = []  
    test_images.append(test_img)
    test_images = np.array(test_images)
    img_rows, img_cols = 150, 150
    test_images = test_images.reshape(-1, img_rows * img_cols, 1)
    test_images = test_images.astype('float32')
    test_images /= 255
    face_pred = face_model.predict(test_images)[0][0]
    return face_pred

def detect_video_body(img, body_model):
    test_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    test_img = cv2.resize(test_img, (150, 150))
    test_img = np.array(test_img)
    test_images = []  
    test_images.append(test_img)
    test_images = np.array(test_images)
    img_rows, img_cols = 150, 150
    test_images = test_images.reshape(-1, img_rows * img_cols, 1)
    test_images = test_images.astype('float32')
    test_images /= 255
    body_pred = body_model.predict(test_images)[0][1]
    return body_pred

def detect_video(filename = "predict/df_video1.mp4"):
    face_model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=None,
                         use_skip_connections=True,
                         use_batch_norm = True)
    face_model.summary()
    face_model.load_weights('checkpoints/image_checkpoint.hdf5')

    body_model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=150*150,
                         use_skip_connections=True,
                         use_batch_norm = True)
    body_model.summary()
    body_model.load_weights('checkpoints/body_lang_checkpoint.hdf5')

    counter = 1
    real_face_frames = 0
    df_face_frames = 0
    real_body_frames = 0
    df_body_frames = 0

    vid = cv2.VideoCapture(filename)
    while True:
        _, img = vid.read()
        if(counter%5!=0):
            counter+=1
            continue
        if img is None:
            print("Empty Frame")
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img)
        pil_image = Image.fromarray(img)

        if face_locations == 0:
            continue

        body_pred = detect_video_body(img,body_model)
        print("Confidence of it being a deepfake based on body language:", body_pred)
        if(body_pred<0.5):
            real_body_frames += 1
        else:
            df_body_frames += 1

        for face_location in face_locations:
            top, right, bottom, left = face_location
            test_img = pil_image.crop((left-50,top-50,right+50,bottom+50))
            
            face_pred = detect_video_face(test_img, face_model)
            print("Confidence of it being a deepfake based on face:", face_pred)
            if(face_pred<0.5):
                real_face_frames += 1
            else:
                df_face_frames += 1
        print(counter, "frames detected.")
        counter += 1
  
    print("Number of real face frames: ", real_face_frames)
    print("Number of deepfake face frames: ", df_face_frames)
    print("Number of real body frames: ", real_body_frames)
    print("Number of deepfake body frames: ", df_body_frames)
    return real_face_frames, df_face_frames, real_body_frames, df_body_frames

if __name__ == '__main__':
    detect_video(filename = "predict/df_video.mp4")
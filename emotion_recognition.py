# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:55:58 2020

@author: #PSP
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:55:58 2020

@author: #PSP
"""

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition


#Capturing from web cam
webcam_video_stream = cv2.VideoCapture(0)

#Loading the training dataset
face_exp_model = model_from_json(open("dataset/face_model.json", "r").read())
face_exp_model.load_weights("dataset/face_model.h5")
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

all_face_location = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx = 0.25, fy = 0.25)
    all_face_location = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample = 2, model = "hog")
    
    for index, current_face_location in enumerate(all_face_location):
    
        #Splitting tuple to get the four positions values
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        
        #Printing the faces coordinates
        print('Found face {} at top : {}, right: {}, bottom: {}, left: {}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
        
        #Slicing the image
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        
        #converting to grayscale image
        #current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        
        #Converting the 48 X 48 px size
        #current_face_image = cv2.resize(current_face_image, (48, 48))
        
        #Converting PIL (Returned by pillow library) image to 3d numpy array
        #img_pixels = image.img_to_array(current_face_image)
        
        #Expanding array to single row multiple columns
        #img_pixels = np.expand_dims(img_pixles, axis = 0)
        
        #Here in present pixels are in range of 0 to 255 so converting it in range of 0 or 1 coz the gray scale image has only o and 1 value
        #img_pixels /= 255
        
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
        #converting to grayscale image
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        
        #Converting the 48 X 48 px size
        current_face_image = cv2.resize(current_face_image, (48, 48))
        
        #Converting PIL (Returned by pillow library) image to 3d numpy array
        img_pixels = image.img_to_array(current_face_image)
        
        #Expanding array to single row multiple columns
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        #Here in present pixels are in range of 0 to 255 so converting it in range of 0 or 1 coz the gray scale image has only o and 1 value
        img_pixels /= 255
        
        #Do prediction using model and get the pridiction value for all 7 emotions
        exp_predictions = face_exp_model.predict(img_pixels)
        
        #Find max predicted value in 0 to 7
        max_index = np.argmax(exp_predictions[0])
        
        #Get corresponding label for emotion
        emotion_label = emotions_label[max_index]
        
        #Display the emotion for the user
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
        
        
        
        
    cv2.imshow("Webcam Video", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
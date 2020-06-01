# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:40:38 2020

@author: #PSP
"""


import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

#Loading image
image_to_detect = cv2.imread('images/tm.jpg')

#Loading the training dataset
face_exp_model = model_from_json(open("dataset/face_model.json", "r").read())
face_exp_model.load_weights("dataset/face_model.h5")
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#Detect all faces in image using hog model which is fast
all_faces_locations = face_recognition.face_locations(image_to_detect, model = "hog")

#Detect all faces in image using cnn model which is Slower but accurate about faces angle and inclination too
#all_faces_locations = face_recognition.face_locations(image_to_detect, model = "cnn")


#printing the number of faces are detected
print('There are {} number of faces in image'.format(len(all_faces_locations)))

#Looping through the face locations
for index, current_face_location in enumerate(all_faces_locations):
    
    #Splitting tuple to get the four positions values
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    
    #Printing the faces coordinates
    print('Found face {} at top : {}, right: {}, bottom: {}, left: {}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
    
    #Slicing the current face from image
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
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
    cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
        
        
        
cv2.imshow("Image Face Emotions", image_to_detect)
#cv2.imshow('test', image_to_detect)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

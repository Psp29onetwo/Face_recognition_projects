# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:39:37 2020

@author: #PSP
"""

import cv2
import face_recognition

#Loading image
original_image = cv2.imread('images/testing/tm2.jpg')

#Lodaing the Modi image
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0] # Here [0] means we are assuming that there are only one face in the image

#Lodaing the Trumph image
trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0] # Here [0] means we are assuming that there are only one face in the image

# Saving the encodings
known_face_encoding = [modi_face_encoding, trump_face_encoding]
known_face_names = ['Narendra Modi', 'Donald Trump']

#Loding the image to recognize
image_to_recognize = face_recognition.load_image_file('images/testing/tm2.jpg')

#Detect all faces in image using hog model which is fast
all_faces_locations = face_recognition.face_locations(image_to_recognize, model = "hog")

#Detecting all encodings for all faces
all_faces_encoding = face_recognition.face_encodings(image_to_recognize, all_faces_locations)


#Detect all faces in image using cnn model which is Slower but accurate about faces angle and inclination too
#all_faces_locations = face_recognition.face_locations(image_to_detect, model = "cnn")


#printing the number of faces are detected
print('There are {} number of faces in image'.format(len(all_faces_locations)))

#Looping through the face locations and face embeddings
for current_face_location, current_face_encoding in zip(all_faces_locations, all_faces_encoding):
    
    #Splitting tuple to get the four positions values
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    
    #Printing the faces coordinates
    #print('Found face {} at top : {}, right: {}, bottom: {}, left: {}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
    
    all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding)
    
    name_of_person = 'Unknown Face'
    
    
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
    cv2.rectangle(original_image, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
    #Showing the current face with dynamic title
    cv2.imshow("Identified Faces", original_image)
    
    
    
    
    
    #Slicing the current face from image
    #current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    
    
#cv2.imshow('test', image_to_detect)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:55:07 2020

@author: #PSP
"""



import cv2 
import face_recognition

#Capturing from web cam
webcam_video_stream = cv2.VideoCapture(0)

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
        #print('Found face {} at top : {}, right: {}, bottom: {}, left: {}'.format(index + 1, top_pos, right_pos, bottom_pos, left_pos))
        
        #Blurring face
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        #current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)
        #current_frame[top_pos:bottom_pos, left_pos:right_pos] = current_face_image
        
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 144.895847746)
        
        #Create the blob for current face
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (244, 244), AGE_GENDER_MODEL_MEAN_VALUES, swapRB = False)
        
        gender_label_list = ["Male", "Female"]
        gender_protext = "dataset/deploy_gender.prototxt"
        gender_caffemodel = "dataset/gender_net.caffemodel"
        
        #Creating model
        gender_convolution_n_network = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        
        #Giving input to the model
        gender_convolution_n_network.setInput(current_face_image_blob)
        
        #Getting the predictions from the model
        gender_predictions = gender_convolution_n_network.forward()
        
        #Finding the maximum value from the predictions
        gender = gender_label_list[gender_predictions[0].argmax()]
        
        age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        
        age_protext = "dataset/deploy_age.prototxt"
        age_caffemodel = "dataset/age_net.caffemodel"
        
        #Creating model
        age_convolution_n_network = cv2.dnn.readNet(age_caffemodel, age_protext)
        
        #Giving input to the model
        age_convolution_n_network.setInput(current_face_image_blob)
        
        #Getting the predictions from the model
        age_predictions = age_convolution_n_network.forward()
        
        #Finding the maximum value from the predictions
        age = age_label_list[age_predictions[0].argmax()]
        
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos + 20), (0, 0, 255), 2) # Here (0, 0, 0) means the border have trancparent colour i.e. excluding the border
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender + " " + age + "yrs", (left_pos, bottom_pos), font, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Webcam Video", current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
import cv2
from model import FacialExpressionModel
import numpy as np
#import requests


#URL = "http://192.168.2.4:8080/video"
#while True:
 ##  img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
   # img = cv2.imdecode(img_arr, -1)

    ##cv2.imshow("AndroidCam", img)


    #if cv2.waitKey(1) == 27:
     #   break


rgb = cv2.VideoCapture(0) # For IPWEBCAM use "http://192.168.2.4:8080/video" instead of 0 in VideoCapture
facec = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face. 
    
    returns: tuple (faces in image, frame read, grayscale frame)
    """
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray

def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1
        
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        imS = cv2.resize(fr, (960, 540))  
        cv2.imshow('Filter', imS)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("dataset/face_model.json", "dataset/face_model.h5")
    start_app(model)
#import the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2
import time

# loading the stored model from file
model=load_model('/home/alaa/Desktop/SOS_FYP/FIre_Detection_only/Fire-64x64-color-v7-soft.h5')
#capture frames from the camera with source ID 0
#cap = cv2.VideoCapture(0)
#give the camera time to heat up
time.sleep(2)
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	orig=image.copy()
        #resize image for better processing speeds
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        #get the probability of the captured frame in percentage
        fire_prob = model.predict(image)[0][0] * 100
        # print("Fire Probability: ", fire_prob)
        # print("Predictions: ", model.predict(image))
        label = "Fire Probability: " + str(fire_prob)
        #write the fire probability on the shown frames 
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        cv2.imshow("Output", orig)
        #wait for keys to exit the window
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
    elif rval==False:
            break
cap.release()
cv2.destroyAllWindows()


import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
while(1):
    rval,img = cap.read()
    if rval==True:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(img_hsv, (0,100,100), (10,255,255))  #CALIB
        mask2 = cv2.inRange(img_hsv, (160,100,100), (179,255,255)) #CALIB
        mask = cv2.bitwise_or(mask1, mask2 )
        croped = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("Output", croped)

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
    elif rval==False:
        break
        end = time.time()


cap.release()
cv2.destroyAllWindows()

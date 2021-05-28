import numpy as np 
from picamera.array import PiRGBArray as RGB 
from picamera import PiCamera 
import time 
import cv2 

camera = PiCamera() 
camera.resolution = (640,480) 
camera.framerate = 30 
rawCapture = RGB(camera, size = (640,480)) 
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
pastTime = 0
elapseTime = 0

time.sleep(0.1) 
   
for frame in camera.capture_continuous(rawCapture, format="bgr",       
          use_video_port=True):                           
    image = cv2.rotate(frame.array,cv2.cv2.ROTATE_180)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, 
                     minNeighbors=5, minSize=(20,20))

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
    elapseTime = time.time()
    fps = 1 / (elapseTime - pastTime)
    pastTime = elapseTime
    cv2.putText(image, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN,3, (255,0,0),3)
    cv2.imshow("VIDEO",image)                                   
    key = cv2.waitKey(1) & 0xFF                         
    rawCapture.truncate(0)                                                  
    if key == ord('q'):
        break 

cv2.destroyAllWindows()

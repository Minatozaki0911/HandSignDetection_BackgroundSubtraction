import numpy as np 
from picamera.array import PiRGBArray as RGB 
from picamera import PiCamera 
import time 
import cv2 
    
camera = PiCamera() 
camera.resolution = (640,480) 
camera.framerate = 30 
rawCapture = RGB(camera, size = (640,480)) 
   
time.sleep(0.1) 
   
for frame in camera.capture_continuous(rawCapture, format="bgr",       
          use_video_port=True):                           
    image = cv2.rotate(frame.array,cv2.cv2.ROTATE_180)
    cv2.imshow("VIDEO",image)                                   
    key = cv2.waitKey(1) & 0xFF                         
    rawCapture.truncate(0)                                                  
    if key == ord('q'):
        break 

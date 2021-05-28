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
    image = frame.array
    cv2.imshow(image)
    key = cv2.waitkey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break


while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1) # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()

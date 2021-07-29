from urllib import request
import cv2
import numpy as np
import time

URL='http://192.168.1.4:8080/shot.jpg'

while True:
    stream = np.array(bytearray(request.urlopen(URL).read()), dtype=np.uint8)
    img = cv2.imdecode(stream, -1)
    img = cv2.resize(img, (720, 400))
    cv2.imshow('Phone Capture', img)

    key = cv2.waitKey(1)
    if key ==ord('q'):
        break

cv2.destroyAllWindows()

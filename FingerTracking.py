import cv2
import time
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

def PiCameraInit():
    camera = PiCamera()
    camera.resolution = (640,480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640,480))
    time.sleep(0.1)

def drawRectangle(frame):
    row, col, _ = frame.shape
    global totalRectangle, rectX1, rectY1, rectX2, rectY2

    rectX1 = np.array([6*row/20, 6*row/20, 6*row/20,
                        9*row/20, 9*row/20, 9*row/20,
                        12*row/20, 12*row/20, 12*row/20],
                        dtype=np.uint32)

    rectY1 = np.array([6*col/20, 6*col/20, 6*col/20,
                        9*col/20, 9*col/20, 9*col/20,
                        12*col/20, 12*col/20, 12*col/20],
                        dtype=np.uint32)

    rectX2 = rectX1 + 10
    rectY2 = rectY1 + 10

    for i in range(totalRectangle):
        cv2.rectangle(frame, (rectY1[i],rectX1[i]),(rectY2[i],rectX2[i]),
                (210,145,188),thickness=1)
    return frame


def handHistogram(frame):
    global handRectX1, handRectY1
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    roi = np.zeros([90,10,3],dtype=hsv.dtype)

    for i in range (totalRectangle):
        roi[i*10 : i*10+10, 0 : 10] = hsv[handRectX1[i]:handRectX1[i]+10, 
                                        handRectY1[i]:handRectY1[i]+10]

    handHist = cv2.calcHist([roi], [0,1],None, [180,256],[0,180,0,256])
    return cv2.normalize(handHist,handHist, 0, 255, cv2.NORM_MINMAX)

def histMasking(frame, hist):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    dst = cv2.calcBackProject(hsv, [0,1],hist, [0,180,0,156],1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh =  cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh = cv2.merge((thresh,thresh,thresh))

    return cv2.bitwise_and(frame, thresh)

def centroid(contour):
    moment = cv2.moments(contour)
    if moment['m00']!=0:
        centroidX = int(moment['m10'] / moment['m00'])
        centroidY = int(moment['m01'] / moment['m00'])
        return centroidX,centroidY
    else:
        return None

def farthestPoint(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:,0][:,0]
        centroidX, centroidY = centroid
        
        x = np.array(contour[s][:,0][:,0], dtype=np.float)
        y = np.array(contour[s][:,0][:,1], dtype=np.float)

        distance = cv2.sqrt(cv2.add(cv2.pow(cv2.subtract(x,centroidX),2),
                                    cv2.pow(cv2.subtract(y,centroidY),2)))
        maxDistanceIndex = np.argmax(distance)

        if maxDistanceIndex < len(s):
            farthestDefects = s[maxDistanceIndex]
            farthestPoint = tuple(contour[farthestDefects][0])
            return farthestPoint
        else:
            return None

def MousePointer(frame, pos):
    if pos is not None:
        for i in range(len(pos)):
            cv2.circle(frame, pos[i], int(5-(5*i*3)/100), [0,255,125],-1)

def rescale(frame, scale):
    w = int(frame.shape[1] * scale / 100)
    h = int(frame.shape[0] * scale / 100)
    return cv2.resize(frame, (w,h), interpolation=cv2.INTER_AREA)

def contours(histMaskImage):
    grayImage = cv2.cvtColor(histMaskImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayImage, 0, 255, 0)
    _, cont, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def calculatePosition(frame, hist):
    histMaskImage = histMasking(frame, hist)
    histMaskImage = cv2.erode(histMaskImage, None, iteration=2)
    histMaskImage = cv2.dilate(histMaskImage, None, iteration=2)
    contourList = contours(histMaskImage)
    maxContour = max(contourList, key = cv2.contourArea)

    contourCentroid = centroid(maxContour)
    cv2.circle(frame, contourCentroid, 5, [150,100,255], -1)

    if maxContour is not None:
        hull = cv2.convexHull(maxContour, returnPoints=False)
        defects = cv2.convexityDefects(maxContour, hull)
        fingertipsPos = farthestPoint(defects, maxContour, contourCentroid)
        print("Centroid: "+str(contourCentroid) + ", fingertips: " + str(fingertipsPos))
        cv2.circle(frame, fingertipsPos, 4, [100, 255, 100], -1)

        if len(positionList) < 20:
            positionList.append(fingertipsPos)
        else:
            positionList.pop(0)
            positionList.append(fingertipsPos)

        MousePointer(frame, positionList)

if __main__ == "__main__":
    PiCameraInit()
    global handHist
    userHandPreference = False

    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        frame = cv2.rotate(frame.array, cv2.cv2.ROTATE_180)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            print("Taking User Hand Preference")
            userHandPreference = True
            handHist = handHistogram(frame)

        if userHandPreference:
            calculatePosition(frame, handHist)

        else:
            frame = drawRectangle(frame)

        cv2.imshow("Result Window", rescale(frame))
        rawCapture.truncate(0)
        
        if key == ord('r'):
            print("Recalibrate")
            userHandPreference = False

        if key == ord('q'):
            print("Closed")
            break
    cv2.destroyAllWindows()

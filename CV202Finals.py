from keras.models import load_model
import cv2
import numpy as np

roiW = 0.35
roiH = 0.55
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
smallkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
minY = 54
maxY = 163
minCr = 130
maxCr = 173
minCb = 76
maxCb = 126
minHue = 0
maxHue = 179 
minSat = 0                                                  
maxSat = 100                                               
minVal = 0                              
maxVal = 150                                                 

prediction = ''
score = 0

gesture_names = {0: 'Nam Tay',
                 1: 'Chu L',
                 2: 'OK',
                 3: 'Chu V',
                 4: 'Ban tay'}
model = load_model('./Model')
print("model loaded")

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

def updateHSV(x):                                         
    global minHue, maxHue, minSat, maxSat, minVal, maxVal
    minHue = cv2.getTrackbarPos('min Hue', 'control')    
    maxHue = cv2.getTrackbarPos('max Hue', 'control')    
    minSat = cv2.getTrackbarPos('min Sat', 'control')    
    maxSat = cv2.getTrackbarPos('max Sat', 'control')    
    minVal = cv2.getTrackbarPos('min Value', 'control')  
    maxVal = cv2.getTrackbarPos('max Value', 'control')  

def updateYCrCb(x):                                         
    global minY, maxY, minCr, maxCr, minCb, maxCb
    minY = cv2.getTrackbarPos('min Y', 'control')    
    maxY = cv2.getTrackbarPos('max Y', 'control')    
    minCr = cv2.getTrackbarPos('min Cr', 'control')    
    maxCr = cv2.getTrackbarPos('max Cr', 'control')    
    minCb = cv2.getTrackbarPos('min Cb', 'control')  
    maxCb = cv2.getTrackbarPos('max Cb', 'control')  

def controlPanel():                                              
    cv2.namedWindow("control")                                   
    cv2.resizeWindow('control', 550, 20)
    cv2.createTrackbar('min Hue', 'control', 0, 179, updateHSV)  
    cv2.createTrackbar('max Hue', 'control', 0, 179, updateHSV)  
    cv2.createTrackbar('min Sat', 'control', 0, 255, updateHSV)  
    cv2.createTrackbar('max Sat', 'control', 0, 255, updateHSV)  
    cv2.createTrackbar('min Value', 'control', 0, 255, updateHSV)
    cv2.createTrackbar('max Value', 'control', 0, 255, updateHSV)
    cv2.createTrackbar('min Y', 'control', 0, 255, updateYCrCb)  
    cv2.createTrackbar('max Y', 'control', 0, 255, updateYCrCb)  
    cv2.createTrackbar('min Cr', 'control', 16, 240, updateYCrCb)  
    cv2.createTrackbar('max Cr', 'control', 16, 240, updateYCrCb)  
    cv2.createTrackbar('min Cb', 'control', 16, 240, updateYCrCb)  
    cv2.createTrackbar('max Cb', 'control', 16, 240, updateYCrCb)  
    print("Init control panel")

AdaptiveGaussianModel = cv2.createBackgroundSubtractorMOG2(history=60, detectShadows=False)
faceCascade = cv2.CascadeClassifier('./haarcascade.xml')

controlPanel()
counter = 0
vid = cv2.VideoCapture('./test.mp4')
while(vid.isOpened()):
    ret, frame = vid.read()
    counter += 1
    if counter == vid.get(cv2.CAP_PROP_FRAME_COUNT):
        counter = 0
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = cv2.resize(frame, (0,0),  fx=0.7, fy=0.7)
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
    faceFrame = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    face = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in face:
        cv2.rectangle(faceFrame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(faceFrame, (int(x+w/2),int(y+h/4)), radius=5, color=(255,0,0), thickness=-1)
        cv2.imshow('Result', faceFrame)

        refPoint = hsv[int(y+h/4)-3:int(y+h/4)+3, int(x+w/2)-3:int(x+w/2)+3, :]
        print("HSV updated ------------------ ")
        refPoint = np.average(refPoint, axis=0)
        minHue = np.average(refPoint, axis=0)[0]-7
        maxHue = np.average(refPoint, axis=0)[0]+5
        minSat = np.average(refPoint, axis=0)[1]-87
        maxSat = np.average(refPoint, axis=0)[1]+3
        minVal = np.average(refPoint, axis=0)[2]-95
        maxVal = np.average(refPoint, axis=0)[2]+60
        print("Min Hue: ", minHue)
        print("Min Sat: ", minSat)
        print("Min Val: ", minVal)
        
        refPoint = ycrcb[int(y+h/4)-3:int(y+h/4)+3, int(x+w/2)-3:int(x+w/2)+3, :]
        print("YCrCb updated -----------------")
        refPoint = np.average(refPoint, axis=0)
        minCr = np.average(refPoint, axis=0)[1]-20
        maxCr = np.average(refPoint, axis=0)[1]-8
        minCb = np.average(refPoint, axis=0)[2]-5
        maxCb = np.average(refPoint, axis=0)[2]+38
        print("Min Cr: ", minCr)
        print("Min Cb: ", minCb)

    lowerhsv = np.array([minHue, minSat, minVal],np.uint8)
    upperhsv = np.array([maxHue, maxSat, maxVal],np.uint8)
    blurMask = cv2.inRange(hsv, lowerhsv, upperhsv)
    mask_3d = np.repeat(blurMask[:, :, np.newaxis], 3, axis=2)
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    blur = np.where(mask_3d == (255,255,255), frame, blurred_frame)
    #cv2.imshow('Blurred Background', blur)

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(HSV, lowerhsv, upperhsv)
    skinMaskHSV = cv2.morphologyEx(skinMaskHSV, cv2.MORPH_OPEN, kernel)
    skinMaskHSV = cv2.morphologyEx(skinMaskHSV, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("HSV Mask", skinMaskHSV)

    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lowerb = np.array([minY, minCr, minCb],np.uint8)
    upperb = np.array([maxY, maxCr, maxCb],np.uint8)
    skinMaskYCrCb = cv2.inRange(YCrCb, lowerb, upperb)
    #cv2.imshow("YCrCb Mask", skinMaskYCrCb)

    skinMask = cv2.bitwise_or(skinMaskHSV, skinMaskYCrCb)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("2 mask combined", skinMask)

    AdaptiveGaussianMask = AdaptiveGaussianModel.apply(frame)
    #cv2.imshow('Background mask', AdaptiveGaussianMask)

    contours, hierarchy = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (255,255,255), thickness=-1)

    hybrid = cv2.bitwise_and(skinMask, skinMask, mask=AdaptiveGaussianMask)
    #cv2.imshow("Color and background mask combined", hybrid)

    contours, hierarchy = cv2.findContours(hybrid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255,255,225), thickness=cv2.FILLED)
    frame  = cv2.morphologyEx(frame, cv2.MORPH_OPEN, smallkernel)
    frame  = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, smallkernel)

    ROI = frame[0:int(roiH*frame.shape[0]), 20:int(roiW*frame.shape[1]+20)]
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(ROI, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", thresh)
    print(thresh.shape) 
    if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0])>0.1):
        if(thresh is not None):
            target = np.stack((thresh,)*3, axis=-1)
            target = cv2.resize(target, (224,224))
            target = target.reshape(1, 224, 224, 3)
            prediction, score = predict_rgb_image_vgg(target)
            print(score, prediction)
            if (score>=90):
                cv2.putText(faceFrame, "Sign:" + prediction, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (250, 0, 250), 2, lineType=cv2.LINE_AA)
                cv2.putText(faceFrame, "Confidence:" + str(score), (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (250, 0, 250), 2, lineType=cv2.LINE_AA)
                cv2.imshow('Result', faceFrame)


    if cv2.waitKey(10)==ord('q'):
        break
    if cv2.waitKey(10)==ord('p'):
        print('Paused')
        while(1):
            key = cv2.waitKey(0)
            if key == ord('p'):
                print('Pause ended')
                break

vid.release()
cv2.destroyAllWindows()

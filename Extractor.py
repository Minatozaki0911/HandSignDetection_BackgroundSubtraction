import cv2 
  
# Function which take path as input and extract images of the video 
def ExtractImages(path): 
      
    # Path to video file --- capture_image is the object which calls read
    capture_image = cv2.VideoCapture(path) 

    #keeping a count for each frame captured  
    frame_count = 0
  
    while (True): 
        #Reading each frame
        con,frames = capture_image.read() 
        #con will test until last frame is extracted
        if con:
            #giving names to each frame and printing while extracting
            name = str(frame_count)+'.jpg'
            print('Cutting  --> '+name)
  
            # Extracting images and saving with name 
            cv2.imwrite(name, frames) 

            frame_count = frame_count + 1
        else:
            break
  
path = r"./Video/six.mp4"

ExtractImages(path)

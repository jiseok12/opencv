import cv2
import numpy as np

CAMERA_ID = 0

cam = cv2.VideoCapture(CAMERA_ID)
def nothing():
    pass

if cam.isOpened() == False:
    print
    'Cannot open camera - %d' % (CAMERA_ID)
    exit()
    
cv2.namedWindow('RGB track bar')
cv2.createTrackbar('red color', 'RGB track bar', 1, 50, nothing)

cv2.setTrackbarPos('red color', 'RGB track bar', 3)

# img = np.zeros((512, 512, 3), np.uint8)


while True:
    ret, frame = cam.read()
    image=frame.copy()
    filter_val = cv2.getTrackbarPos('red color', 'RGB track bar')
    
    
    if filter_val % 2 == 0:
        filter_val += 1
        
    image =cv2.GaussianBlur(image, (filter_val,filter_val), 0)
    cv2.imshow('RGB track bar', image)

    if cv2.waitKey(10) > 0:
        break

cam.release()
cv2.destroyWindow()
import os 
import cv2
cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    seccess, frame = cap.read()
    if seccess:
        cv2.imshow('iamge', frame)
        
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27:
            break
    else:
        break
    
cap.release()

cv2.destroyAllWindows()
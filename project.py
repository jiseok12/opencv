# 차영상 
# 이진화 
import cv2
CAMERA_ID = 0

cam = cv2.VideoCapture(CAMERA_ID)

if cam.isOpened() == False:
    print 
    'Cannot open ther camera -%d' % (CAMERA_ID)
    exit()
    
cv2.namedWindow('CAM Window')
count = 1
count2 = 1
back=None
while(True):
    ret, frame = cam.read()
    
    if not ret:
        break
    key = cv2.waitKey(33)
    if key == ord('a'):
        count = 2

        back = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        back = cv2.GaussianBlur(back, (0, 0), 1.0)
        
    if count == 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        
        frame = cv2.absdiff(gray, back)
        
    if key == ord('b'):
        count2 = 2
        
    if count2 == 2:
        _, frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
    
       # cnt, _, stats, _ = cv2.connectedComponentsWithStats(frame)
    cv2.imshow('CAM Window', frame)
    
    
    if key == ord('q'):
        break
    
    
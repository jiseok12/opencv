import numpy as np
import cv2
from matplotlib import pyplot as plt

count1 = [0]


xyvalue = []

img1 = cv2.imread("./road.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (320, 240))
h, w= img1.shape
cv2.namedWindow('test')
def test(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.rectangle(img1, (x,y), (x+5, y+5), (255, 0,0), -1)
        count1[0] = count1[0]+1
        xyvalue.append([x, y])
cv2.setMouseCallback('test',test)

ress = []    
while(1):
    cv2.imshow('test', img1)
    # print(count1)
    if count1[0] == 4:
        point1_src = np.float32(xyvalue)
        point1_dst = np.float32([[xyvalue[0][0],xyvalue[0][1]], [xyvalue[0][0],xyvalue[1][1]], [xyvalue[3][0],xyvalue[2][1]], [xyvalue[3][0],xyvalue[3][1]]])
        per_mat2 = cv2.getPerspectiveTransform(point1_src, point1_dst)
        res = cv2.warpPerspective(img1, per_mat2, (w, h))
        ress = []
        ress.append(img1)
        ress.append(res)
        
        
        break
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.imshow(ress[i], cmap='gray')
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.destroyWindow()


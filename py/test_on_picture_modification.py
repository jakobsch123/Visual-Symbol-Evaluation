# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:14:03 2020

@author: alehof
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread(r'../img/Test2.jpg',0)
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print("Number of contours = " + str(len(contours)))
print(contours[0])

sorted_ctrs = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[1] + cv.boundingRect(ctr)[0] * img.shape[0] )
i=0
for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv.contourArea(cnt) < 300):
        continue

    # Filtered countours are detected
    x,y,w,h = cv.boundingRect(cnt)

    # Taking ROI of the cotour
    # change this
   # imginvert= cv2.bitwise_not(img)
   # imgray = cv2.cvtColor(imginvert, cv2.COLOR_BGR2GRAY)
   # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    roi = th3[y:y+h, x:x+w]
    roi= cv.bitwise_not(roi)
    # add this
    roi= cv.copyMakeBorder(roi, 10, 10, 10, 10, cv.BORDER_CONSTANT)
    
    # Mark them on the image if you want
   # cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

    # Save your contours or characters
    cv.imwrite("../img/roi" + str(i) + ".png", roi)

    i = i + 1 
 




cv.destroyAllWindows()
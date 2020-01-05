# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:00:34 2020

@author: alehof
"""
# Promising Links
# https://stackoverflow.com/questions/52995607/how-to-segment-handwritten-and-printed-digit-without-losing-information-in-openc/53081313
# https://gist.github.com/pknowledge/af2cfb0753abe45c563188794773618f
# https://answers.opencv.org/question/179510/how-can-i-sort-the-contours-from-left-to-right-and-top-to-bottom/



import numpy as np
import cv2

img = cv2.imread(r'../img/Sample5.png')
imginvert= cv2.bitwise_not(img)
imgray = cv2.cvtColor(imginvert, cv2.COLOR_BGR2GRAY)
#imginvert = cv2.bitwise_not(imgray)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Number of contours = " + str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

cv2.imshow('Image', img)
cv2.imshow('Image GRAY', imgray)

orig = img.copy()
i = 0


#sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
# Sorting from Left to Right, doesn't take in mind multiple possible rows
sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[0] * img.shape[0] )
#sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )
# x + y * w




for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv2.contourArea(cnt) < 100):
        continue

    # Filtered countours are detected
    x,y,w,h = cv2.boundingRect(cnt)

    # Taking ROI of the cotour
    roi = imgray[y:y+h, x:x+w]
    
    #roi= cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT)

    # Mark them on the image if you want
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

    # Save your contours or characters
    cv2.imwrite("../img/roi" + str(i) + ".png", roi)

    i = i + 1 

cv2.imshow("Image", orig) 




cv2.destroyAllWindows()
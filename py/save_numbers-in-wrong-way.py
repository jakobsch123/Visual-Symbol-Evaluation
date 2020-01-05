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
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1] )
# x + y * w




for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv2.contourArea(cnt) < 100):
        continue

    # Filtered countours are detected
    x,y,w,h = cv2.boundingRect(cnt)

    # Taking ROI of the cotour
    roi = img[y:y+h, x:x+w]

    # Mark them on the image if you want
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

    # Save your contours or characters
    cv2.imwrite("../img/roi" + str(i) + ".png", roi)

    i = i + 1 

cv2.imshow("Image", orig) 




cv2.destroyAllWindows()
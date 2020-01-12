import io
import pyqrcode
from base64 import b64encode

import numpy as np
import urllib
import requests
import matplotlib as plt

import cv2 as cv
import eel

eel.init('web')


@eel.expose
def dummy(dummy_param):
    print("I got a parameter: ", dummy_param)
    return "string_value", 1, 1.2, True, [1, 2, 3, 4], {"name": "eel"}


@eel.expose
def generate_qr(data):
    img = pyqrcode.create(data)
    buffers = io.BytesIO()
    img.png(buffers, scale=8)
    encoded = b64encode(buffers.getvalue()).decode("ascii")
    print("QR code generation successful.")
    return "data:image/png;base64, " + encoded

@eel.expose
def numberofcontours(img):
	print(img)
	img = cv.imread(img,0)
	img = cv.medianBlur(img,5)
	ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# 	th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,11,2)
	th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv.THRESH_BINARY,11,2)
# 	titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# 	images = [img, th1, th2, th3]
# 	for i in range(4):
# 		pyplot.subplot(2,2,i+1),pyplot.imshow(images[i],'gray')
# 		pyplot.title(titles[i])
# 		pyplot.xticks([]),pyplot.yticks([])

	contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	print("Number of contours = " + str(len(contours)))
	print(contours[0])

	sorted_ctrs = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[1] + cv.boundingRect(ctr)[0] * img.shape[0] )
	i=0
	for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
	    if(cv.contourArea(cnt) < 150):
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
	return i

eel.start('basic3.html', size=(1000, 600))

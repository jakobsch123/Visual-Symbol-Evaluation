# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:24:30 2019

@author: jakob
"""
# example of loading the mnist dataset
#from keras.datasets import mnist
#from matplotlib import pyplot

# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import GaussianNoises

from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(GaussianNoise(0.1))
    model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(model, dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['acc'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()

	# evaluate model
	scores, histories = evaluate_model(model, trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
	
	# serialize model to JSON
	model_json = model.to_json()
	with open("modelv3.json", "w") as json_file:
		json_file.write(model_json)    
    # serialize weights to HDF5
	model.save_weights("modelv3.h5")
	print("Saved model to disk")

def load_existing_model():
	# load json and create model
	json_file = open('modelv3.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("modelv3.h5")
	print("Loaded model from disk")
	
	# evaluate loaded model on test data
	a, b, X, Y = load_dataset()
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	score = model.evaluate(X, Y, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
	model
	return model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
	pyplot.imshow(img)
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

def numberofcontours():
	img = cv2.imread(r'../img/test44.png')
	imginvert=cv2.bitwise_not(img)
	imgray = cv2.cvtColor(imginvert, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	#thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	print("Number of contours = " + str(len(contours)))
	print(contours[0])
	#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	#cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

	#cv2.imshow('Image', img)
	#cv2.imshow('Image GRAY', imgray)

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
		imginvert= cv2.bitwise_not(img)
		   # imgray = cv2.cvtColor(imginvert, cv2.COLOR_BGR2GRAY)
		# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		roi = imginvert[y:y+h, x:x+w]
	      # add this
		roi= cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

		    # Taking ROI of the cotour
		#roi = img[y:y+h, x:x+w]
		#roi = imgray[y:y+h, x:x+w]

		    # Mark them on the image if you want
		cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

		    # Save your contours or characters
		cv2.imwrite("../img/roi" + str(i) + ".png", roi)
		i = i + 1
	cv2.destroyAllWindows()
	return i

def predict_image_with_existing_model(numofpics):
	# load model
	model = load_existing_model()
	for i in range(numofpics):
	# load the image
		#img = load_image('../img/fuenf_handwritten.png')
		img = load_image('../img/roi'+ str(i) + '.png')
		# predict the class
		digit = model.predict_classes(img)
		print(digit[0])

# entry point, run the test harness
run_test_harness()
#predict_image_with_existing_model(21)
#predict_image_with_existing_model(numberofcontours())
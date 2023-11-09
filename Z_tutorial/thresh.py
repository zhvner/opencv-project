import cv2 as cv
import matplotlib.pylab as plt 
import numpy as np 

img = cv.imread('Photos/cats.jpg')
cv.imshow('cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Graysclae', gray)

# simple thresholding  - binarazation of an image convert it to binary image, where 
# pixels are 0 - black or 255 - white
# take an image, take particular value of thresh - compare each pixel val to this threshold
# if below this thresh - then set to black - 0
# if above - then set to white - 1

# simple thresholding 

threshold, thresh = cv.threshold(gray, thresh=150, maxval=255, type=cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)
# lower the thresh value, more colors are white
# higher the thresh value, more colors are black

# simple thresholding  - inverse
adaptive_thresh = cv.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, 
                                       thresholdType=cv.THRESH_BINARY, blockSize=11, C=3)
# blockSize=11 is neighborhood size of kernel size to find optimal threshold value
# C=3 substracted from the mean
cv.imshow('Adaptive thresholding', adaptive_thresh)

# -- inverse
adaptive_thresh_inv = cv.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, 
                                       thresholdType=cv.THRESH_BINARY_INV, blockSize=11, C=3)

cv.imshow('Adaptive thresholding inverse', adaptive_thresh_inv)

# simple thresholding  - inverse 

cv.waitKey(0)
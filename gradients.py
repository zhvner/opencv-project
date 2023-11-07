import cv2 as cv
import matplotlib.pylab as plt 
import numpy as np 

img = cv.imread('Photos/cats.jpg')
cv.imshow('cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Graysclae', gray)

# gradients and edges are completely different in math terms
# canny edge detector = advanced edge detection algorithm - multistep process 

# Laplacian
# when you transition from b to w or vice versa, it's either +/- slope
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap)) # converted to absolute values then to uint8 (img specific datatype)
cv.imshow('Laplacian', lap)
# looks like pencil shading

# Sobel  - gradient magnitude representation
sob_x = cv.Sobel(gray, cv.CV_64F, 1,0)
sob_y = cv.Sobel(gray, cv.CV_64F, 0,1) 
combined_sobel = cv.bitwise_or(sob_x, sob_y)

cv.imshow('Sobel X', sob_x)
cv.imshow('Sobel Y', sob_y)
cv.imshow('Sobel Combined', combined_sobel)

# compare with canny - advanced algorithm , uses sobel  in one of its stages
# multi stage algorithm where one of the stages includes sobel to calcualte gradients
canny = cv.Canny(gray, threshold1=150, threshold2=175)
cv.imshow('Canny', canny)

cv.waitKey(0)
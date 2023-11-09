# we need blurring when there's noise happening in the picture

import cv2 as cv
import numpy as np 

img = cv.imread('Photos/park.jpg')

cv.imshow('Park', img)

# there's a window that we choose anfd it has a size and it has kernel size
# 3x3 - kernel size
# blur is applied to middle pixel - 2 x 2
# averaging - compute the middle pixel as the avg of the surrounding pixels

average = cv.blur(img, (3,3))
cv.imshow('Average blur',average)

average = cv.blur(img, (7,7))
cv.imshow('Average blur',average)
# way more blur

#gaussian blur - each surrounding pixel is given a particular weight - more natural,
# gives a true value for the center pixel

gaussian = cv.GaussianBlur(img, (7,7),sigmaX=0) # sigmaX=0 is sd
cv.imshow('Gaussian blur',gaussian)

# less blurred than average (7,7)

# Median blur - works the same as averaging but instead of average of surrounding pixels it finds median 
medianBlur = cv.medianBlur(img, 7 )
# no need to provide a tuple for kernel size, just say 3
cv.imshow('Median blur',medianBlur)
# very blurred

# bilateral blur - the most effective, retains the edges 
bilateral = cv.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=25)
cv.imshow('Bilaterial blur',bilateral)
# by setting sigmaSpace=15, we show that we can go to the further distance to calculate center pixel of kernel size


cv.waitKey(0)
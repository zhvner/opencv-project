import cv2 as cv
import numpy as np 

img = cv.imread('Photos/cats.jpg')
cv.imshow('cats', img)

# figures
blank = np.zeros(img.shape[:2], 'uint8')
 # circle = cv.circle(blank.copy(), (200,200),200,255,-1)


circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', circle)
rect = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
weird_shape=cv.bitwise_and(circle,rect)
cv.imshow('Weird AND',weird_shape)

masked1 = cv.bitwise_and(img, img, mask=circle)
masked_weird = cv.bitwise_and(img, img, mask=weird_shape)

cv.imshow('Masked image circle', masked1)
cv.imshow('Masked image weird', masked_weird)

# masking2 = cv.rectangle(blank, (img.shape[1]//2, img.shape[0]//2), (img.shape[1]//2+100, img.shape[0]//2+100), 255, -1)
# cv.imshow('Mask', masking2)

# masked2 = cv.bitwise_and(img, img, mask=masking2)

# cv.imshow('Masked image rect', masked2)




cv.waitKey(0)
import cv2 as cv
import numpy as np 

img = cv.imread('Photos/cats.jpg')

cv.imshow('cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

canny = cv.Canny(img, 125, 175)
cv.imshow('Edges of Gray', canny)

# takes in edges = canny, mode = cv.RETR_LIST - all the contours int the image
# RETR_TREE - hirarchal contours, cv.RETR_EXTERNAL - external contours
#  cv.CHAIN_APPROX_NONE - contour approximation method - HOW? - does nothing, just returns all contours (different endpoint coordinates)
#  cv.CHAIN_APPROX_SIMPLE is preferred - compresses all the contours returned (2 endpoint coordinates)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# contours - python list of all the coordinates of contours that were found in the image
# hierarchies - hierarchal represenaation of contours e.g. you have rectangle -> inside square, inside circle

print(f'{len(contours)} contours found!')

contours1, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours1)} contours found!')

cv.drawContours(image=blank, contours=contours1, contourIdx=-1, color=(0,0,255) , thickness=1)
cv.imshow('Contours drawn', blank)


canny = cv.Canny(blur, 125, 175)
cv.imshow('Edges of Gray', blur)

#type=cv.THRESH_BINARY - binarizing the image
ret, thresh = cv.threshold(gray, 125, 255, type=cv.THRESH_BINARY)

contours1, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours1)} contours found!') # 380 contours found! - reduced bc of blur
cv.imshow('Thresh Image', thresh)

# 839 contours found! - for threshold image

cv.waitKey(0)
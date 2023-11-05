import cv2 as cv

img = cv.imread('Photos/park.jpg')
cv.imshow('Park',img)

#Converting to gray scale 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blur 
blur1 = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blurred1', blur1)

blur2 = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blurred2', blur2)

# edge cascade 
canny = cv.Canny(blur1, 125, 175)
cv.imshow('Canny', canny)

# dilated 
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('Dilated', dilated)
# not much difference
dilated = cv.dilate(canny, (3,3), iterations=3)
cv.imshow('Dilated', dilated)

# eroding 
eroded = cv.erode(dilated, (3,3), iterations=4)
cv.imshow('Eroded', eroded)

# resize 
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

# cropping the image
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

#if 0 is passed in the argument it waits till any key is pressed 
cv.waitKey(0)
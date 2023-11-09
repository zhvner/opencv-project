import cv2 as cv
import numpy as np 

img = cv.imread('Photos/park.jpg')

cv.imshow('Park', img)

def translate(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x = left
# -y = up
# x = right
# y = down

translated = translate(img, -100, -100)
cv.imshow('Translated', translated)


def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle,(1.0))
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 90)
cv.imshow('Rotated',rotated)

rotated = rotate(img, -90)
cv.imshow('Rotated clockwise',rotated)

# resizing 
resized = cv.resize(img, (500,500),interpolation=cv.INTER_CUBIC)
# INTER_CUBIC if enlarging the image
cv.imshow('Resized',resized)

resized = cv.resize(img, (1000,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)


# flip an image
flipped = cv.flip(img, 0)
cv.imshow('Flipped',flipped)


flipped_hor = cv.flip(img, 1)
cv.imshow('Flipped Horizontal',flipped_hor) # mirror 

# cropping 
cropped = img[200:300, 300:400]
cv.imshow('Cropped',cropped)



cv.waitKey(0)
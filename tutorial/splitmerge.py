import cv2 as cv
import numpy as np 

img = cv.imread('Photos/park.jpg')

cv.imshow('Park', img)
blank = np.zeros(img.shape[:2], dtype = 'uint8')
b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank])
red = cv.merge([blank,blank ,r])
green = cv.merge([blank,g, blank])



cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)
# darker means no color in that area

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red) 

print(img.shape) #(427, 640, 3)
print(b.shape) #(427, 640)
print(g.shape) #(427, 640)
print(r.shape)#(427, 640)


cv.waitKey(0)
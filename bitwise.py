import cv2 as cv
import numpy as np 

blank = np.zeros((400,400),'uint8')
rect = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200),200,255,-1)

cv.imshow('rect',rect )
cv.imshow('circ',circle )

# AND - internsection
bitwise_and = cv.bitwise_and(rect, circle)
cv.imshow('AND', bitwise_and)

# OR - union - superpose
bitwise_or = cv.bitwise_or(rect, circle)
cv.imshow('OR', bitwise_or)

# XOR - non intersection
bitwise_xor = cv.bitwise_xor(rect, circle)
cv.imshow('XOR', bitwise_xor)

# NOT - inverts the binary color
bitwise_not = cv.bitwise_not(rect, circle)
cv.imshow('NOT', bitwise_not)

cv.waitKey(0)
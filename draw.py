import cv2 as cv
import numpy as np 

#(500,500,3) = (width, height, number of color channels)
blank = np.zeros((500,500,3), dtype = 'uint8')
cv.imshow('Blank', blank)

# - display cat image 
# img = cv.imread('Photos/cat.jpg')
# cv.imshow('Cat',img)

# - paint the image a certain color
# blank[200:300, 300:350] = 0,255,0
# cv.imshow('Green', blank)

# - draw a rectangle 
# cv.rectangle(blank,(0,0),(250,250),(0,255,0), thickness=2)
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0), thickness=2)

# thickness negative =  fills the rectange or cv.FILLED 

# - draw a circle 
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40, (0,0,255), thickness=3)
cv.circle(blank,(blank.shape[1]//4,blank.shape[0]//3),50, (0,0,255), thickness=-1)


# - draw a circle  
cv.line(blank,(0,0),(blank.shape[1]//3,blank.shape[0]//2),(255,255,255), thickness=3)


# - write text on the image 
cv.putText(blank, 'Hello, I am Zhanerke', (255,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0,0))
cv.imshow('With Text', blank)
cv.waitKey(0)

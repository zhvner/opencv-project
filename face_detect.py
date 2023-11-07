import cv2 as cv

img = cv.imread('Photos/lady.jpg')
cv.imshow('Lady', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Graysclae', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml') # - reads
# cv.CascadeClassifier (xml file) reads the xml trained data
cv.waitKey(0)
import cv2 as cv

img = cv.imread('Photos/lady.jpg')
cv.imshow('Lady', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Graysclae', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml') # - reads
# cv.CascadeClassifier (xml file) reads the xml trained data

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3 )
print(f'number of faces found = {len(faces_rect)}')
# number of faces found = 1

# - draw a rectangle
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('detected face', img)



# group 2
group =  cv.imread('Photos/group 2.jpg')
gray_group = cv.cvtColor(group, cv.COLOR_BGR2GRAY)
cv.imshow('Gray group 2', gray_group)
# faces_rect = haar_cascade.detectMultiScale(gray_group, scaleFactor=1.1, minNeighbors=3 )

# shows 7 ()additionally stomachs as faces
#  but true val is 5, it is because of the noise in the image
# -- so we increase min neighbours to 7
faces_rect = haar_cascade.detectMultiScale(gray_group, scaleFactor=1.1, minNeighbors=6)
print(f'number of faces found = {len(faces_rect)}')
# --- 
for (x,y,w,h) in faces_rect:
    cv.rectangle(group, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Group', group)


# group 1
group1 =  cv.imread('Photos/group 1.jpg')
gray_group1 = cv.cvtColor(group1, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray_group1, scaleFactor=1.1, minNeighbors=1)
print(f'number of faces found = {len(faces_rect)}')
# number of faces found = 14 with k=3
# number of faces found = 19 with k=1
# --- so by minimizing k, we are more prone to noise
for (x,y,w,h) in faces_rect:
    cv.rectangle(group1, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    # some faces are not chosen bc they are not perpendicular to the camera

cv.imshow('Group 1', group1)

cv.waitKey(0)
import numpy as np
import cv2 as cv

# reading xml, npy,yml files
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


# testing
img = cv.imread(r'/Users/janner/tutorial/Faces/val/madonna/4.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# detect the face on the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                label, confidence = face_recognizer.predict(faces_roi)
                print(f'Label = {people[label]} with confidence {confidence}')
                cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255, 0), thickness=2)

cv.imshow('Detected face', img)

cv.waitKey(0)

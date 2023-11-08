import cv2 as cv
import numpy as np
import os 





# method 2
# p=[]

# for i in os.listdir(r'Faces/train'):
#     p.append(i)
# print(p)



# loop over every folder in Faces\train
# inside the folder -> it will loop over every image 
# grab the face of that image -> add that to our training set

# method 1

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = (r'Faces/train')
haar_cascade = cv.CascadeClassifier('haar_face.xml') # - reads

features =[] # image array of faces
labels = [] # whose face it belongs to

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
        
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')
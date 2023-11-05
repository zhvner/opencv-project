import cv2 as cv

# img = cv.imread('Photos/cat_large.jpg')
# cv.imshow('Cats', img)




# reading videos

capture = cv.VideoCapture('Videos/dog.mp4') # 0 - webcam , 1-first camera connected to your computer int or path

while True:
    isTrue, frame = capture.read() # grab the video frame by frame
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF ==ord('d'): # if the letter d is pressed, then break out of this loop 
        break

capture.release()
cv.destroyAllWindows()
cv.waitKey(0)
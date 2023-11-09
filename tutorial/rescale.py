import cv2 as cv

img = cv.imread('Photos/cat.jpg')



def rescaleFrame(frame, scale=0.75):
    # frame.shape[1] - frame width
    # frame.shape[0] - frame height
    # convert to int()
    # works for Images, Video and Live video
    width = int(frame.shape[1] * scale )
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# changing resolution of videos
def changeRes(width, height):
    # works only for live video
    capture.set(3, width)
    capture.set(4, height)


resized_image1 = rescaleFrame(img)
cv.imshow('Resized Cat1', resized_image1)
resized_image2 = rescaleFrame(img, scale=0.2)
cv.imshow('Resized Cat2', resized_image2)


capture = cv.VideoCapture('Videos/dog.mp4') # 0 - webcam , 1-first camera connected to your computer int or path

while True:
    isTrue, frame = capture.read() # grab the video frame by frame
    frame_resized = rescaleFrame(frame, scale =0.2)

    # cv.imshow('Video', frame)
    # cv.imshow('Video_resized', frame_resized)

    if cv.waitKey(20) & 0xFF ==ord('d'): # if the letter d is pressed, then break out of this loop 
        break


capture.release()
cv.destroyAllWindows()
cv.waitKey(0)
from ultralytics import YOLO
import cv2 as cv

# Create a VideoCapture object for the default camera = webcam (camera index 0)
capture = cv.VideoCapture(0)

# getting frame height/width from webcam object
frame_width = int(capture.get(3))
frame_height= int(capture.get(4))

# 'output.avi' - This is the name of the output video file
# : cv.VideoWriter_fourcc('M', 'J', 'P', 'G') -
# This specifies the video codec to be used for encoding the video. In this example, it's using the MJPEG codec.
#  MJPEG is a common choice for video encoding.
# 10: This is the frames per second (FPS)
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
model = YOLO('../YOLO-weights/yolov8n.pt')

while True:
    success, img =capture.read() #reading frame by frame
    cv.imshow("Webcam", img)
    if cv.waitKey(0) & 0xFF == ord('1'):
        break
out.release()
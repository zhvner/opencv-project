from ultralytics import YOLO
import cv2 as cv

model = YOLO('yolov8n.pt')
results = model('../Images/Photos/group 1.jpg', show=True)
cv.waitKey(0)


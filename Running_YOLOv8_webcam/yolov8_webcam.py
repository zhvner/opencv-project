from ultralytics import YOLO
import cv2 as cv
import math

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

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = capture.read() #reading frame by frame
    # Detection frame by frame
    # stream=True -
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y2, x2, y2)
            cv.rectangle(img, (x1, y2),(x2, y2), (255, 0, 255), 3)
            #  to round the confidence score (or confidence value) associated with a detected object
            conf = math.ceil((box.conf[0] * 100)) / 100
            # accesses the class label associated with the detected object. The class label represents the category
            # or class to which the detected object belongs. It is often stored as a numerical value.
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv.getTextSize(label, 0, 1, 2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv.LINE_AA)  # filled
            cv.putText(img, label,
                       (x1, y1 - 2),
                       0,
                       1,
                       [255, 255, 255],
                       1,
                       cv.LINE_AA)
    out.write(img)
    cv.imshow("Webcam", img)
    if cv.waitKey(1) & 0xFF == ord('1'):
        break
out.release()
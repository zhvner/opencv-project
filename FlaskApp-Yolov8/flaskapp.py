from flask import Flask, Response, jsonify, request

import cv2 as cv
from YOLO_video import video_detection
app = Flask(__name__)
app.config['SECRET KEY'] = 'zhvner'

def generate_frames (path_x=''):
    yolo_output = video_detection(path_x)
    for detected in yolo_output:
        ref, buffer = cv.imencode('.jpg', detected)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame +
              b'\r\n')
@app.route('/video')

def video():
    return Response(generate_frames(path_x='../Videos/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)



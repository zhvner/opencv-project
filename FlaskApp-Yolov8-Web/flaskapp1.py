from flask import Flask, Response, jsonify, request, session, render_template
import cv2 as cv
from YOLO_video import video_detection

#  gain an intuitive and convenient way to create, validate, and manage web forms within their Flask projects.
from flask_wtf import FlaskForm

from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
# -- explained
# "FileField" for handling file uploads
# "SubmitField" for submit buttons
#  "StringField" for text input,
# "DecimalRangeField" for decimal number input within a specified range
# "IntegerRangeField" for integer number input within a specified range.

from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
#  enabling access to operating system-dependent functionality
# -- with file paths, system configurations,
# and other operating system-related operations.
# os=operating system
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'zhvner'
app.config['UPLOAD_FOLDER'] = 'static/files'

#getting input video and confidence value from user
class UploadFileForm(FlaskForm):
    # "File" path in the filefield
    # validators - ensuring correct format of video
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")



def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detected in yolo_output:
        ref, buffer = cv.imencode('.jpg', detected)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detected in yolo_output:
        ref, buffer = cv.imencode('.jpg', detected)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])

@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')

# @app.route('/video', methods=['GET', 'POST'])
# def video():
#     session.clear()
#     return render_template('video.html')

# @app.route('/webcam',methods=['GET', 'POST'])
# def webcam():
#     session.clear()
#     return render_template('ui.html')

@app.route('/FrontPage',methods=['GET', 'POST'])
def front():
    # instance  to handle file uploads.
    form = UploadFileForm()
    # if a form has been submitted and passes validation
    if form.validate_on_submit():
        #  form is valid ->  line retrieves the data associated with the file input field
        # ->  assigns it to the variable file
        file = form.file.data

        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        # secure_filename(file.filename) -  used to sanitize filenames and remove any potentially unsafe characters.
        # os.path.join(...) -  join the obtained directory path + 'UPLOAD_FOLDER' +  secure filename to create the
        # complete path
        # Use session storage to save video file path -- to save the file to the specified path.
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('video.html', form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x = 0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=8000)
import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Video Capture through webcam
capture = cv2.VideoCapture(0)

# History, Threshold, DetectShadows
fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

def generate_frames():
    while True:
        # Return Value and the current frame
        ret, frame = capture.read()

        # Check if a current frame actually exists
        if not ret:
            break

        # Resize the frame
        resizedFrame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Get the foreground mask
        fgmask = fgbg.apply(resizedFrame)

        # Count all the non zero pixels within the mask
        count = np.count_nonzero(fgmask)

        if count > 5000:
            alert_message = 'Someone stealing your things'
        else:
            alert_message = ''

        _, buffer = cv2.imencode('.jpg', resizedFrame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

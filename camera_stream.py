from jetbot import Camera
from flask import Flask, Response
import cv2

camera = Camera.instance(width=300, height=300)

app = Flask(__name__)

def gen_frames():
    while True:
        frame = camera.value
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)

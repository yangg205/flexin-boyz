#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from flask import Flask, Response
import numpy as np
import cv2

app = Flask(__name__)
latest_frame = None

def image_callback(msg):
    global latest_frame
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    # Nếu encoding là rgb8
    img = arr.reshape(msg.height, msg.width, 3)
    latest_frame = cv2.imencode('.jpg', img)[1].tobytes()

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if latest_frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    rospy.init_node('camera_stream_node')
    rospy.Subscriber('/csi_cam_0/image_raw', Image, image_callback)
    app.run(host='0.0.0.0', port=5001)

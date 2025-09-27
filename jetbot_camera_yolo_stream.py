# jetbot_camera_yolo_stream.py
from flask import Flask, Response
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)

def gstreamer_pipeline(capture_width=640, capture_height=480,
                       display_width=640, display_height=480,
                       framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# Load YOLO model
model_path = "/home/jetbot/flexin-boyz/model/best.onnx"
model = YOLO(model_path)
print("[INFO] YOLO model loaded.")

# Camera
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Không mở được camera CSI.")

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # YOLO detection
        results = model(frame)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                name = model.names[cls_id]
                # Vẽ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{name}:{conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Encode jpeg
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

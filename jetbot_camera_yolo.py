# jetbot_camera_yolo.py
import cv2
from ultralytics import YOLO
import time

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

def main():
    # Load YOLO model (.onnx nhẹ cho Nano)
    model_path = "/home/jetbot/flexin-boyz/best.onnx"  # Thay đường dẫn nếu khác
    model = YOLO(model_path)
    print("[INFO] YOLO model loaded.")

    # Mở camera CSI
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Không mở được camera CSI.")
        return
    print("[INFO] Camera CSI opened successfully.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize nếu cần
            img = cv2.resize(frame, (640, 640))

            # Predict
            results = model(img)

            # Log kết quả
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    name = model.names[cls_id]
                    print(f"[DETECTION] {name}: {conf:.2f}")

            time.sleep(0.05)  # 20 fps
    except KeyboardInterrupt:
        print("[INFO] Đang thoát...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()

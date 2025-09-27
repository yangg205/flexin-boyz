from jetbot import Camera, Robot
from IPython.display import display, Image, clear_output
import cv2
from ultralytics import YOLO
import time

# Khởi tạo camera CSI và robot
camera = Camera.instance(width=300, height=300)
robot = Robot()

# Load YOLO model (thay bằng đường dẫn model của bạn)
model_path = 'best.pt'
model = YOLO(model_path)

try:
    while True:
        # Lấy frame từ camera
        img = camera.value
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Chạy YOLO
        results = model(img_bgr)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Chuyển ảnh về JPEG để hiển thị trong notebook
        _, jpeg = cv2.imencode('.jpg', img_bgr)
        display(Image(data=jpeg.tobytes()))
        clear_output(wait=True)  # xóa frame trước để update frame mới
        
        time.sleep(0.05)  # ~20 FPS
except KeyboardInterrupt:
    camera.stop()
    robot.stop()

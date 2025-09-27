import cv2
from IPython.display import display, Image, clear_output
from jetbot import Camera
import time
import numpy as np

# Khởi tạo camera CSI
try:
    camera = Camera.instance(width=300, height=300)
except Exception as e:
    print("Không mở được camera CSI:", e)

# Hàm hiển thị video trong notebook
def show_frame(frame):
    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Encode to PNG
    _, buf = cv2.imencode('.png', rgb_frame)
    display(Image(data=buf.tobytes()))

# Vòng lặp hiển thị video
try:
    for _ in range(100):  # Chạy 100 frame
        frame = camera.value
        clear_output(wait=True)  # Xóa frame cũ
        show_frame(frame)
        time.sleep(0.05)
finally:
    camera.stop()

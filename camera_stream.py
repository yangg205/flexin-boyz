import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Khởi tạo YOLO
model = YOLO('/home/jetbot/flexin-boyz/model/best.pt')
# Camera
camera = cv2.VideoCapture(0)

# ASCII chars
ASCII_CHARS = "@%#*+=-:. "

# Resize frame cho terminal
TERM_WIDTH = 80
TERM_HEIGHT = 40

def frame_to_ascii(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (TERM_WIDTH, TERM_HEIGHT))
    ascii_frame = []
    for row in small:
        line = "".join([ASCII_CHARS[pixel * len(ASCII_CHARS) // 256] for pixel in row])
        ascii_frame.append(line)
    return ascii_frame

def overlay_boxes_ascii(ascii_frame, results, frame_shape):
    """Đánh dấu các box YOLO trên ASCII frame"""
    h_ratio = TERM_HEIGHT / frame_shape[0]
    w_ratio = TERM_WIDTH / frame_shape[1]
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * w_ratio)
            x2 = int(x2 * w_ratio)
            y1 = int(y1 * h_ratio)
            y2 = int(y2 * h_ratio)
            for y in range(max(0, y1), min(TERM_HEIGHT, y2)):
                line = list(ascii_frame[y])
                for x in range(max(0, x1), min(TERM_WIDTH, x2)):
                    line[x] = "#"
                ascii_frame[y] = "".join(line)
    return ascii_frame

while True:
    ret, frame = camera.read()
    if not ret:
        continue

    results = model(frame)  # YOLO detect
    ascii_frame = frame_to_ascii(frame)
    ascii_frame = overlay_boxes_ascii(ascii_frame, results, frame.shape)

    os.system('clear')  # xóa terminal cũ
    print("\n".join(ascii_frame))
    time.sleep(0.05)  # fps ~20

from jetbot import Camera
from pyzbar import pyzbar
import cv2
import numpy as np

camera = Camera()

while True:
    image = camera.value  # Lấy ảnh từ camera
    if image is not None:
        # Chuyển ảnh từ RGB sang BGR để dùng với OpenCV
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Quét mã QR
        decoded_objects = pyzbar.decode(frame)
        for obj in decoded_objects:
            print('QR Code data:', obj.data.decode('utf-8'))
            # Vẽ khung quanh mã QR trên ảnh
            pts = obj.polygon
            pts = np.array([[pt.x, pt.y] for pt in pts], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

        # Hiển thị ảnh ra màn hình
        cv2.imshow('JetBot QR Scan', frame)
        if cv2.waitKey(1) == ord('q'):
            break

camera.stop()
cv2.destroyAllWindows()

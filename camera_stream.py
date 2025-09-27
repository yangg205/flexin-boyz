from jetbot import Camera
import cv2

camera = Camera.instance(width=300, height=300)

while True:
    frame = camera.value  # numpy array
    # ví dụ convert sang BGR nếu cần YOLO
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # --- chạy YOLO ---
    # results = model(frame_bgr)  # nếu model đã load
    # for r in results:
    #     do something

    # hiển thị terminal (headless chỉ in info, không dùng cv2.imshow)
    print("Got frame:", frame_bgr.shape)

import cv2
from IPython.display import display, clear_output
import ipywidgets as widgets

# Mở camera CSI
cap = cv2.VideoCapture(0)  # /dev/video0
if not cap.isOpened():
    raise RuntimeError("Không mở được camera CSI")

# Widget hiển thị video
image_widget = widgets.Image(format='jpeg')
display(image_widget)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', frame_rgb)
        image_widget.value = jpeg.tobytes()

        clear_output(wait=True)
        display(image_widget)

except KeyboardInterrupt:
    print("Dừng stream camera")

finally:
    cap.release()  # giải phóng camera

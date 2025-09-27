import cv2
from IPython.display import display, clear_output
import ipywidgets as widgets

# GStreamer pipeline cho camera CSI
gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Không mở được camera CSI")

image_widget = widgets.Image(format='jpeg')
display(image_widget)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, jpeg = cv2.imencode('.jpg', frame)
        image_widget.value = jpeg.tobytes()

        clear_output(wait=True)
        display(image_widget)

except KeyboardInterrupt:
    print("Dừng stream camera")

finally:
    cap.release()

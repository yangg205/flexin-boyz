from IPython.display import display, Image, clear_output
import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        continue
    _, jpeg = cv2.imencode('.jpg', frame)
    display(Image(data=jpeg.tobytes()))
    clear_output(wait=True)

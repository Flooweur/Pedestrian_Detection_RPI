from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

cv2.startWindowThread()

model = YOLO('yolov8n.pt')
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

print("Starting the detection")
while True:
    img = picam2.capture_array()
    results = model(source=img)
    annoted_img = results[0].plot()
    cv2.imshow("Detection", annoted_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
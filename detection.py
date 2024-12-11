from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

cv2.startWindowThread()

model = YOLO('yolov8n.pt')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

print("Starting the detection")
while True:
    img = picam2.capture_array()
    results = model.predict(source=img, show=True)
    print(results)


cap.release()
cv2.destroyAllWindows()
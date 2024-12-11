from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import os
cv2.startWindowThread()

model = YOLO('yolov8n.pt')
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

output_dir = "detections"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

print("Starting the detection")
while True:
    img = picam2.capture_array()
    results = model(source=img)
    annoted_img = results[0].plot()
    cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", annoted_img)
    frame_count += 1
    print("Frame count: ", frame_count)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Detection finished")
cv2.destroyAllWindows()
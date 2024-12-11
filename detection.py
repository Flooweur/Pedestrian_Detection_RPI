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

fps = 30
gst_pipeline = (
    "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast "
    "key-int-max=60 ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5000"
)

video_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, (1280, 720))

if not video_writer.isOpened():
    print("Video writer not opened")
    exit(1)

print("Starting the detection")
try:

    while True:
        img = picam2.capture_array()
        results = model(source=img)
        annoted_img = results[0].plot()
        print("Frame count: ", picam2.frame_count)
        video_writer.write(annoted_img)
except KeyboardInterrupt:
    print("Keyboard interrupt")
finally:
    picam2.stop()
    video_writer.release()

print("Detection finished")
cv2.destroyAllWindows()
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
from flask import Flask, Response

app = Flask(__name__)
model = YOLO('yolov8n.pt')

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 360)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.camera_controls["FrameRate"] = 30
picam2.configure("fast")
picam2.start()

last_frame_time = 0
fps_limit = 10

def generate_frames():
    global last_frame_time
    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1.0/fps_limit:
            continue
        last_frame_time = current_time
        img = picam2.capture_array()
        results = model(source=img)
        annoted_img = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annoted_img)
        frame = buffer.tobytes()
        print("frame")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

### You can donate at https://www.buymeacoffee.com/mmshilleh 

app = Flask(__name__)

#model = YOLO('training_output/runs/detect/train/weights/model_pruned.pt')
model = YOLO('training_output/runs/detect/train/weights/best.pt')

camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 360)}, controls={"FrameRate": 60}))
camera.configure("fast")
camera.start()


def generate_frames():
    while True:
        frame = camera.capture_array()
        
        results = model(source=frame)
        annoted_img = results[0].plot()
        
        ret, buffer = cv2.imencode('.jpg', annoted_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

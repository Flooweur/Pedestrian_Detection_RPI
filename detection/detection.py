from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import argparse
import cv2


app = Flask(__name__)

# model = YOLO('training_output/runs/detect/train/weights/best_quantized.onnx')

# camera = Picamera2()
# camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 360)}, controls={"FrameRate": 60}))
# camera.configure("fast")
# camera.start()

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
    parser = argparse.ArgumentParser(description="Pipeline to export, pre-process and quantize a YOLOv8n model.")
    parser.add_argument("--input_model", type=str, required=True, help="YOLOv8n weights path as input (ex: yolov8n.pt)")

    args = parser.parse_args()
    
    model = YOLO(args.input_model)

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 360)}, controls={"FrameRate": 60}))
    camera.configure("fast")
    camera.start()

    app.run(host='0.0.0.0', port=5000)
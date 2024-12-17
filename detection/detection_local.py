from ultralytics import YOLO
import cv2
from flask import Flask, Response
import time
import argparse

app = Flask(__name__)
# Load the YOLO model
# model = YOLO('training_output/runs/detect/train/weights/model_pruned.pt')
# model = YOLO('training_output/runs/detect/train/weights/best_quantized.onnx')
# model = YOLO('training_output/runs/detect/train/weights/best.pt')

# # Initialize webcam
# webcam = cv2.VideoCapture(0)  # Use 0 for the default webcam, or adjust for other cameras

# # Set webcam properties (optional)
# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# webcam.set(cv2.CAP_PROP_FPS, 60)

last_frame_time = 0
fps_limit = 60

def generate_frames():
    global last_frame_time
    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1.0 / fps_limit:
            continue
        last_frame_time = current_time

        # Read frame from the webcam
        ret, img = webcam.read()
        if not ret:
            print("Failed to capture image from webcam")
            break

        # Apply YOLO model
        results = model(source=img)
        annoted_img = results[0].plot()

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', annoted_img)
        frame = buffer.tobytes()

        # Generate the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Pipeline to export, pre-process and quantize a YOLOv8n model.")
        parser.add_argument("--input_model", type=str, required=True, help="YOLOv8n weights path as input (ex: yolov8n.pt)")

        args = parser.parse_args()
        
        # Initialize webcam
        model = YOLO(args.input_model)
        webcam = cv2.VideoCapture(0)  # Use 0 for the default webcam, or adjust for other cameras

        # Set webcam properties (optional)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        webcam.set(cv2.CAP_PROP_FPS, 60)

        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        webcam.release()  # Release the webcam resource when the script stops

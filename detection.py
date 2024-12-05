from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Load model

cap = cv2.VideoCapture(0)  # Webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, show=True)  # Inference
    cv2.imshow("Pedestrian Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
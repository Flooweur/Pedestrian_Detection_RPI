from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

# Exports the trained weights to onnx:
# model = YOLO("training_output/runs/detect/train/weights/best.pt")
# model.export(format='onnx')

# Quantize the model
#model_input = "training_output/runs/detect/train/weights/best.onnx"
model_input = "preprocessed.onnx"
model_output = "training_output/runs/detect/train/weights/best_quantized.onnx"
quantize_dynamic(model_input, model_output, weight_type=QuantType.QUInt8)

import subprocess
import time
import argparse
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_model_to_onnx(input_file):
    """Step 1: Export model YOLOv8 to ONNX format."""

    print("Step 1: Export model YOLOv8 to ONNX format...")

    model = YOLO(input_file)
    model.export(format='onnx')

    print(f"Model exported as '{input_file.split('.pt')[0]}.onnx'.")


def preprocess_onnx_model(input_file):
    """Step 2: Pre-processing of the ONNX model for quantization."""

    print("Step 2: Pre-processing of the ONNX model for quantization...")

    command = [
        "../venv/Scripts/python", "-m", "onnxruntime.quantization.preprocess",
        "--input", f"{input_file.split('.pt')[0]}.onnx", 
        "--output", "preprocessed.onnx"
    ]

    subprocess.run(command, check=True)

    print("Pre-processing finished. Model saved as 'preprocessed.onnx'.")


def quantize_model(output_file):
    """Step 3: Dynamic quantization of the ONNX model."""

    print("Step 3: Dynamic quantization of the ONNX model...")

    model_fp32 = 'preprocessed.onnx'

    quantize_dynamic(model_fp32, output_file, weight_type=QuantType.QUInt8)

    print("Quantification terminée. Modèle sauvegardé en 'quantized.onnx'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to export, pre-process and quantize a YOLOv8n model.")
    parser.add_argument("--input_model", type=str, required=True, help="YOLOv8n weights path as input (ex: yolov8n.pt)")
    parser.add_argument("--quantized_path", type=str, required=True, help="Filename after quantization (ex: dynamic_quantized.onnx)")

    args = parser.parse_args()
    start_time = time.time()
    
    try:
        export_model_to_onnx(args.input_model)
        preprocess_onnx_model(args.input_model)
        quantize_model(args.quantized_path)
    except subprocess.CalledProcessError as e:
        print(f"Error when executing the following commande : {e}")
    except Exception as e:
        print(f"An unexpected error happened : {e}")
    else:
        print("Exportation and quantization pipeline successfully finished !")
    finally:
        end_time = time.time()
        print(f"Total duration: {end_time - start_time:.2f} seconds.")

import time
import argparse
import torch, torch.nn as nn
from torch.nn.utils import prune
from ultralytics import YOLO


def sparsity(model):
    '''Return global model sparsity'''
    a = 0
    b = 0

    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()

    return b / a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to prune a YOLOv8n model.")
    parser.add_argument("--input_model", type=str, required=True, help="YOLOv8n weights path as input (ex: yolov8n.pt)")
    parser.add_argument("--output_path", type=str, required=True, help="Filename after pruning (ex: pruning.onnx)")

    args = parser.parse_args()
    start_time = time.time()
    
    try:
        model = torch.load(args.input_model)
        pruning_param = 0.3

        for name, m in model['model'].named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
                prune.l1_unstructured(m, name='weight', amount=pruning_param)
                prune.remove(m, 'weight')

        print(f"Model pruned to {sparsity(model['model']):.3g} global sparsity")

        torch.save(model, args.output_path)
    except Exception as e:
        print(f"An unexpected error happened : {e}")
    else:
        print("Pruning pipeline successfully finished !")

import subprocess
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt').to(device)

command = [
    "yolo", "task=detect", "mode=train", "model=yolov8n.pt"
    "data=../dataset/data.yaml", "epochs=200", "imgsz=640"
    "batch=16", "device=0"
]

subprocess.run(command, check=True)

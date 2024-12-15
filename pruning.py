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


model = torch.load("training_output/runs/detect/train/weights/best.pt")
pruning_param = 0.3

for name, m in model['model'].named_modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
        prune.l1_unstructured(m, name='weight', amount=pruning_param)  # prune
        prune.remove(m, 'weight')  # make permanent

print(f"Model pruned to {sparsity(model['model']):.3g} global sparsity")

torch.save(model, 'training_output/runs/detect/train/weights/model_pruned.pt')

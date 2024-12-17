from ultralytics import YOLO
import torch
import torch.nn as nn

# Load YOLOv3-Tiny (teacher)
teacher_model = YOLO("yolov3-tiny.pt")
print(teacher_model)

# Load YOLOv8n (student)
student_model = YOLO("yolov8n.pt")
print(student_model)

# Set both models to evaluation mode
teacher_model.model.eval()
student_model.model.train()


from your_dataset import PedestrianDataset  # Replace with your dataset class

# Create DataLoader
train_dataset = PedestrianDataset("path_to_dataset", transform=your_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=your_collate_fn)

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, ground_truth):
        # Soft Target Loss
        soft_loss = self.mse_loss(
            torch.softmax(student_logits / self.temperature, dim=-1),
            torch.softmax(teacher_logits / self.temperature, dim=-1)
        )
        
        # Hard Target Loss
        hard_loss = self.ce_loss(student_logits, ground_truth)

        # Combine Losses
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


distillation_loss = DistillationLoss(alpha=0.7, temperature=4.0)
optimizer = optim.Adam(student_model.model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass through teacher and student
        with torch.no_grad():
            teacher_outputs = teacher_model.model(images)
        
        student_outputs = student_model.model(images)

        # Compute distillation loss
        loss = distillation_loss(student_outputs, teacher_outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

student_model.model.eval()
# Perform evaluation using your validation dataset

student_model.export(format="onnx", path="yolov3-tiny-distilled.onnx")



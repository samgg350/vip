from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # downloads automatically on first run
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
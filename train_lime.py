"""Train YOLO on the lime dataset from collect_lime_data.py and copy the result to best.pt."""
import shutil
from ultralytics import YOLO

EPOCHS = 30     # ~10-20 min on a laptop CPU for ~200 images; lower if short on time
IMGSZ = 416     # same size obj_track_advpid.py uses

model = YOLO("yolo26n.pt")  # start from the pretrained model already in this folder
model.train(data="dataset/data.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=16,
            project="runs", name="lime", exist_ok=True, patience=10)

shutil.copy(model.trainer.best, "best.pt")
print(f"Saved best.pt (from {model.trainer.best}). Next: python obj_track_advpid.py")

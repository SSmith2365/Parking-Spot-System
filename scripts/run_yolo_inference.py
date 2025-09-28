from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8l.pt")
img_dir = "label_batch_01_yolo/images"
output_dir = "inference_output_v8l"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(img_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(img_dir, filename)
        results = model(img_path, classes=[2])
        results[0].save(filename=os.path.join(output_dir, filename))

print(f"Inference Complete. Check the {output_dir}/ folder.")

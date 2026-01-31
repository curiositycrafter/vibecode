from ultralytics import YOLO
model = YOLO("yolov11n.pt")  # Will auto-download if available
print(model.names)

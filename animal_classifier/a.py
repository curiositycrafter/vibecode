import os
import shutil
import random
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
# Step 1: Preprocessing and Dataset Preparation
# Assume dataset root path (change if needed)
dataset_root = 'dataset/dataset'  # Folder with subfolders for each class

# Get list of classes (151 animal species)
classes = sorted(os.listdir(dataset_root))
num_classes = len(classes)  # Should be 151
class_map = {cls: idx for idx, cls in enumerate(classes)}  # Map class name to ID

# Collect all image paths
image_paths = []
labels = []
for cls in classes:
    cls_path = os.path.join(dataset_root, cls)
    imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
    image_paths.extend(imgs)
    labels.extend([class_map[cls]] * len(imgs))

# Ensure ~30 imgs per class (total ~4500)
assert len(image_paths) > 4000, "Dataset size mismatch; check download."

# Split into train/val (80/20)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

# Create YOLO dataset structure
yolo_root = 'yolo_animals_151/'
os.makedirs(yolo_root, exist_ok=True)
for split in ['train', 'val']:
    os.makedirs(os.path.join(yolo_root, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, split, 'labels'), exist_ok=True)

# Function to copy images and create dummy labels (full-image box: class 0.5 0.5 1.0 1.0)
def prepare_split(imgs, lbls, split):
    for img_path, lbl in zip(imgs, lbls):
        # Copy image
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(yolo_root, split, 'images', img_name))
        
        # Create label txt (YOLO format: class_id center_x center_y width height normalized)
        label_path = os.path.join(yolo_root, split, 'labels', img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            f.write(f"{lbl} 0.5 0.5 1.0 1.0\n")  # Dummy full box

prepare_split(train_imgs, train_labels, 'train')
prepare_split(val_imgs, val_labels, 'val')

# Create data.yaml for YOLO
data_config = {
    'path': os.path.abspath(yolo_root),
    'train': 'train/images',
    'val': 'val/images',
    'nc': num_classes,
    'names': classes
}
with open(os.path.join(yolo_root, 'data.yaml'), 'w') as f:
    yaml.dump(data_config, f)

print("Dataset prepared with dummy bounding boxes. Note: This assumes single centered animal per image.")

# Step 2: Training YOLOv11
# Load pre-trained YOLOv11 model (use 'yolov11n.pt' for nano/lightweight, or 'yolov11s.pt' for small)
model = YOLO('yolov8n.pt')  # Start with nano for small dataset; change to larger if more compute

# Train with augmentations enabled (default in Ultralytics: flip, mosaic, etc.)
results = model.train(
    data=os.path.join(yolo_root, 'data.yaml'),
    epochs=10,  # Adjust based on convergence; small dataset may need fewer
    imgsz=224,  # Input size; images will be resized
    batch=8,   # Adjust based on GPU memory
    name='animal_detection_151',
    augment=True,  # Enable augmentations for better generalization
    patience=10,   # Early stopping if no improvement
    device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)

# Step 3: Inference Example (Test on a sample image)
# After training, load best weights
model = YOLO('runs/detect/animal_detection_151/weights/best.pt')

# Test on a validation image (replace with your path)
test_img = val_imgs[0]  # First val image
results = model(test_img)

# Print results (includes bounding boxes and class)
results.show()  # Displays image with outlines and labels
print(results)  # Prints detections

print("Training complete. Model saved in 'runs/detect/animal_detection_151/'.")
import os
dataset_path = 'archive/animal_data'
classes = os.listdir(dataset_path)
print(f"Number of classes: {len(classes)}")
print("Classes:", classes)
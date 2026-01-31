import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 15  # Number of animal classes
DATASET_PATH = 'archive/animal_data'

# Explicit class list to ensure correct order
classes = ['Beetle', 'Butterfly', 'Cat', 'Cow', 'Dog', 'Elephant', 'Gorilla', 
           'Hippo', 'Lizard', 'Monkey', 'Mouse', 'Panda', 'Spider', 'Tiger', 'Zebra']

def prepare_validation_data():
    # Create data generator for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=classes,  # Explicitly specify classes
        subset='validation',
        shuffle=False  # Ensure consistent evaluation
    )

    print("Validation class indices:", validation_generator.class_indices)
    print("Number of validation classes:", len(validation_generator.class_indices))
    
    return validation_generator

def evaluate_model():
    # Load the trained model
    model = tf.keras.models.load_model('animal_classifier.h5')
    
    # Prepare validation data
    validation_generator = prepare_validation_data()
    
    # Evaluate model
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == '__main__':
    evaluate_model()
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('animal_classifier.h5')
labels = ['Beetle', 'Butterfly', 'Cat', 'Cow', 'Dog', 'Elephant', 'Gorilla', 'Hippo', 'Lizard', 'Monkey', 'Mouse', 'Panda', 'Spider', 'Tiger', 'Zebra']
IMG_SIZE = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save uploaded image
            uploads_dir = os.path.join(app.root_path, 'static/uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            image_path = os.path.join(uploads_dir, file.filename)
            file.save(image_path)

            # Preprocess and predict
            img = image.load_img(image_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100  # As percentage

            prediction = f"{predicted_class} ({confidence:.2f}%)"
            image_path = f"/static/uploads/{file.filename}"  # For display

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
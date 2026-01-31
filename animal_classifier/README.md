# 🐾 Animal Classifier – Deep Learning Web App

A custom-trained deep learning web application that classifies animal images into 15 categories using a fine-tuned MobileNetV2 model, deployed with Flask.

Built with a mix of structured ML engineering and a little bit of vibe coding ⚡

---

## 🚀 What This Project Does

* Upload an animal image through a web interface
* Runs inference using a trained deep learning model
* Returns predicted class with confidence percentage
* Fully local deployment using Flask

---

## 🧠 Model Architecture

* Base Model: **MobileNetV2 (ImageNet pretrained backbone)**
* Fine-tuned on a custom 15-class animal dataset
* Input Size: 224×224
* Output: 15 softmax categories
* Final trained weights saved as `animal_classifier.h5`

This is not just a pretrained drop-in — the network was trained and adapted for this specific classification task.

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* Flask
* NumPy
* HTML (Jinja templating)

---

## 📂 Project Structure

```
animal-classifier/
│
├── web.py
├── animal_classifier.h5
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── images/
```

---

## ⚙️ How To Run Locally

### 1️⃣ Clone

```bash
git clone <your-repo-link>
cd animal-classifier
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run App

```bash
python web.py
```

Visit:

```
http://127.0.0.1:5000/
```

---

## 📸 Supported Classes

Beetle, Butterfly, Cat, Cow, Dog, Elephant, Gorilla, Hippo, Lizard, Monkey, Mouse, Panda, Spider, Tiger, Zebra

---

## 🎯 Why This Project Matters

* Demonstrates transfer learning
* Shows full ML pipeline: training → saving → deployment
* Integrates deep learning model with a web interface
* Clean reproducible environment setup
* Combines experimentation with execution (a little vibe coding included)

---

from flask import Flask, render_template, request
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only (optional)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model("healthy_vs_rotten.h5")

# Warm-up model (optional)
dummy_input = np.zeros((1, 224, 224, 3))
model.predict(dummy_input)

labels = ["Biodegradable", "Recyclable", "Trash"]

@app.route('/')
def home():
    return render_template('waste.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        result = labels[predicted_class]

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

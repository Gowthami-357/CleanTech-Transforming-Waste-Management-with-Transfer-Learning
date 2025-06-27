# would tou like to exectute this code then you need to give some commands in the terminal
#cd C:\Users\DELL\Desktop\project\municipal_waste_flask\w_flask; `
#if (-Not (Test-Path ".venv\Scripts\Activate.ps1")) { python -m venv .venv }; `
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; `
#.venv\Scripts\Activate.ps1; `
#pip install --upgrade pip; `
#pip install Flask Pillow numpy tensorflow
#python app.py]
#copy all this commands at once(remove the comments in each line) and paste it in the terminal,press enter then you will get a link(http://127.0.0.1:5000) and copy that and paste it in browser
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = load_model('healthy_vs_rotten.h5')  # replace with your actual model path

# Label mapping (update if your model uses different classes)
labels = {0: 'Biodegradable', 1: 'Recyclable', 2: 'Trash'}

# Real prediction function
def predict_label(file_path):
    img = image.load_img(file_path, target_size=(224, 224))  # match your model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    return labels[predicted_class]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    label = predict_label(file_path)
    image_url = f"/static/uploads/{filename}"

    return render_template('result.html', label=label, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)

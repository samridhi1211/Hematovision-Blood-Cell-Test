from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = load_model("model.h5")

classes = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    return render_template('index.html', prediction=result, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
🧬 HematoVision: Blood Cell Classification System

🔍 Overview
HematoVision is a deep learning-based web application that classifies blood cell images into four categories using Transfer Learning. The model is deployed using Flask, allowing users to upload an image and get real-time predictions.

🚀 Features

    📤 Upload blood cell images via web interface
    
    🤖 Deep Learning model (MobileNetV2)
    
    🎯 Real-time prediction
    
    📊 Confidence score display
    
    🖼️ Image preview before prediction
    
    🌐 Flask-based deployment

    
🧠 Model Details

     Architecture: MobileNetV2 (Transfer Learning)
     
     Input Size: 224 × 224
     
     Classes:
           Eosinophil
           Lymphocyte
           Monocyte
           Neutrophil
           
     Training Accuracy: ~80%

     
🛠️ Tech Stack

    Python
    
    TensorFlow / Keras
    
    Flask
    
    HTML / CSS / JavaScript

    
📁 Project Structure


│Hematovision/
├── app.py              # Flask application
├── train.py           # Model training script
├── check_data.py      # Dataset verification
├── model.h5           # Trained model (ignored in GitHub)
│
├── templates/
│   └── index.html     # UI
│
├── static/
│   └── uploads/       # Uploaded images
│
├── dataset/           # Dataset (ignored)
├── .gitignore
└── README.md


▶️ How to Run the Project

  1️⃣ Clone Repository
git clone https://github.com/samridhi1211/Hematovision-Blood-Cell-Test.git
cd Hematovision-Blood-Cell-Test

  2️⃣ Setup Environment
conda create -n hemato python=3.10
conda activate hemato

  3️⃣ Install Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow flask pillow

  4️⃣ Run Application
python app.py

  5️⃣ Open in Browser
http://127.0.0.1:5000/

📸 Demo
  👉 Upload an image → Click Predict → View result


<img width="1855" height="870" alt="Screenshot 2026-03-27 233909" src="https://github.com/user-attachments/assets/3884ad0e-a0f7-48e5-91c6-0efb4fa940ef" />


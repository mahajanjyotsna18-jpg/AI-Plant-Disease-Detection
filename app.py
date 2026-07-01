from flask import Flask, request, send_from_directory
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load AI Model
model = tf.keras.models.load_model("plant_disease_model.h5")

class_names = [
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_blight"
]

solutions = {
    "Tomato_Healthy": "Plant is healthy ✅",
    "Tomato_Early_Blight": "Use fungicide and remove infected leaves.",
    "Tomato_Late_blight": "Remove infected leaves immediately and apply fungicide."
}

# Prediction Function
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        image = request.files["image"]

        filepath = os.path.join(UPLOAD_FOLDER, image.filename)

        image.save(filepath)

        result = predict_image(filepath)
        solution = solutions[result]

        return f"""
        <h2>Prediction Result 🌱</h2>

       <h3>Disease: {result}</h3>
       <h4>Solution:</h4>
<p>{solution}</p>

     <img src="/uploads/{image.filename}" width="300">
        """

    return """
    <h1>AI Plant Disease Detection 🌱</h1>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image">
        <button type="submit">Upload Image</button>
    </form>
    """

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.utils import img_to_array, img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your actual model architecture
pretrained_model = tf.keras.applications.Xception(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

pretrained_model.trainable = False

model2 = tf.keras.Sequential(
    [
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(8, activation="softmax"),
    ]
)

# Load the weights
model2.load_weights("model2_best_weights.h5")
model2.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Classes for prediction
classes = [
    "Mug Beer",
    "Mountain Dew Diet",
    "Mountain Dew Original",
    "Pepsi Cherry",
    "Pepsi Original",
    "Pepsi Real Sugar",
    "Pepsi Zero",
    "Pepsi Diet",
]

stats = defaultdict(lambda: defaultdict(int))


# Helper function to update statistics
def update_stats(class_name, success):
    today = datetime.now().strftime("%Y-%m-%d")
    stats[today][class_name] += success


UPLOAD_FOLDER = "uploads"  # Create a folder named 'uploads' in your project directory

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Function for image preprocessing
def prepare_image(file):
    # Ensure the 'uploads' directory exists
    uploads_folder = "uploads"
    os.makedirs(uploads_folder, exist_ok=True)

    # Save the file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(uploads_folder, filename)
    file.save(filepath)

    # Load the saved file using load_img
    img = load_img(filepath, target_size=(224, 224))

    # Delete the temporarily saved file
    os.remove(filepath)

    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)


# Route for home
@app.route("/")
def home():
    return render_template("index.html")


# Route for predicting class probabilities
@app.route("/predict", methods=["POST"])
def predict():
    images = request.files.getlist("images")
    if not images:
        return render_template("prediction_error.html", error="No images provided"), 400

    predictions = {}
    for img in images:
        img_array = prepare_image(img)
        result = model2.predict(img_array)[0]

        # Round probabilities to four decimals
        class_probs = {
            classes[i]: round(float(result[i]), 4) for i in range(len(classes))
        }
        predicted_class = max(class_probs, key=class_probs.get)

        # Update statistics
        update_stats(predicted_class, success=1)

        # Improved user-friendly output
        predictions[img.filename] = {
            "predicted_class": predicted_class,
            "probabilities": {k: f"{v:.4f}" for k, v in class_probs.items()},
        }

    return render_template("prediction_result.html", predictions=predictions)


# Route for retrieving supported classes
@app.route("/info", methods=["GET"])
def get_supported_classes():
    supported_classes = [
        "Mug Beer",
        "Mountain Dew Diet",
        "Mountain Dew Original",
        "Pepsi Cherry",
        "Pepsi Original",
        "Pepsi Real Sugar",
        "Pepsi Zero",
        "Pepsi Diet",
    ]
    return render_template("info_result.html", supported_classes=supported_classes)


# Route for retrieving statistics
@app.route("/stats", methods=["GET"])
def get_stats():
    last_10_days = [
        (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)
    ]
    result = {day: stats[day] for day in last_10_days}

    return render_template("stats_result.html", stats=result)


@app.route("/<name>")
def user(name):
    return f"This is the /{name} path"


if __name__ == "__main__":
    app.run(debug=False)

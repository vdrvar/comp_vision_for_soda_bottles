from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img
import tensorflow as tf
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict
from shutil import copyfileobj
from pathlib import Path
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import List

app = FastAPI()

# Load your actual model architecture
pretrained_model = tf.keras.applications.Xception(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
pretrained_model.trainable = False

model2 = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(8, activation="softmax"),
])

# Ensure model architecture is fully defined here
model2.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load weights
model2.load_weights("model2_best_weights.h5")

classes = [
    "Mug Beer", "Mountain Dew Diet", "Mountain Dew Original",
    "Pepsi Cherry", "Pepsi Original", "Pepsi Real Sugar", "Pepsi Zero", "Pepsi Diet"
]

stats = defaultdict(lambda: defaultdict(int))

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

def update_stats(class_name, success):
    today = datetime.now().strftime("%Y-%m-%d")
    stats[today][class_name] += success

def prepare_image(file_path):
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

@app.get("/", response_class=HTMLResponse, name="read_root")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse, name="predict")
async def predict(request: Request, images: List[UploadFile] = File(...)):
    predictions = {}
    for img in images:
        file_path = UPLOAD_FOLDER / secure_filename(img.filename)
        with open(file_path, "wb") as buffer:
            copyfileobj(img.file, buffer)
        
        img_array = prepare_image(file_path)
        os.remove(file_path)
        result = model2.predict(img_array)[0]
        class_probs = {classes[i]: round(float(result[i]), 4) for i in range(len(classes))}
        predicted_class = max(class_probs, key=class_probs.get)
        update_stats(predicted_class, success=1)

        predictions[img.filename] = {
            "predicted_class": predicted_class,
            "probabilities": {k: f"{v:.4f}" for k, v in class_probs.items()},
        }
    return templates.TemplateResponse("prediction_result.html", {"request": request, "predictions": predictions})

@app.get("/info", response_class=HTMLResponse, name="get_supported_classes")
async def get_supported_classes(request: Request):
    return templates.TemplateResponse("info_result.html", {"request": request, "supported_classes": classes})

@app.get("/stats", response_class=HTMLResponse, name="get_stats")
async def get_stats(request: Request):
    last_10_days = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)]
    result = {day: stats[day] for day in last_10_days}
    return templates.TemplateResponse("stats_result.html", {"request": request, "stats": result})

def secure_filename(filename):
    return Path(filename).name  # Basic sanitization

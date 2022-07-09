import os
import io

from fastapi import FastAPI, File
from PIL import Image

from model import BearClassifier

app = FastAPI()
classif = BearClassifier(os.path.join(os.path.dirname(__file__), 'models', 'BearClassifier.h5'))

@app.post('/predict')
async def predict_bear(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    label = classif.predict(image)[0].tolist()
    return {"Prediction": label}


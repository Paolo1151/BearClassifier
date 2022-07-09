import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

class BearClassifier():
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, input_image: Image):
        width, height = input_image.size
        resized_image = input_image.resize((224, 224))
        img_np = np.asarray(resized_image)
        img_np = img_np.reshape((1, 224, 224, 3))
        return self.model.predict(img_np)   

    
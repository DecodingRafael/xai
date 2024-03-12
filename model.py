from pathlib import Path

import joblib
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras import backend as K


class Model:
    def __init__(self):
        resnet50_path: Path = Path("../data/resnet50_model.h5")
        model_path: Path = Path("../data/28_09_2023_svm_final_model.pkl")
        Model_Path = model_path
        K.set_learning_phase(0)
        self.resnet_model = load_model(resnet50_path)

        # Load the final model
        self.svm_final = joblib.load(Model_Path)
        self.input_size = (224, 224)

    def extract_features(self, img):
        preprocess_input = resnet50_preprocess_input(img)
        return self.resnet_model.predict(preprocess_input)

    def run_on_batch(self, x):
        test_image_features = self.extract_features(x)
        predictions = self.svm_final.predict_proba(test_image_features)
        return np.array(predictions)

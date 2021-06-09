# 데이터셋 필요, 모델 필요

from tensorflow.keras.models import load_model
from utils.datasets import DataManager
from utils.datasets import DataGenerator

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

input_shape = (128, 128, 3)
verbose = 1

emotion_model_path = 'trained_models/emotion_models/affectnet_mini_XCEPTION_128.h5'
emotion_classifier = load_model(emotion_model_path)

losses = {
    "predictions_exp": "categorical_crossentropy",
    "predictions_etc": "mean_squared_error",
}

dataset_name = 'affect'

# loading dataset
data_loader = DataManager(dataset_path='./datasets/AffectNet/', target_size=input_shape[:2], rescale=1.0 / 255.0)
val_images, val_exp, val_etc = data_loader.get_val_dataset()

emotion_classifier.evaluate(
    val_images, {'predictions_exp': val_exp, 'predictions_etc': val_etc})

# 데이터셋 필요

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from models.cnn import mini_XCEPTION
from utils.datasets import DataManager
from utils.datasets import DataGenerator
import os


batch_size = 256
num_epochs = 10000
input_shape = (224, 224, 3)
verbose = 1

patience = 50
base_path = './trained_models/emotion_models/'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_classes = 8 # Neutral Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt // nf 같은거 없음
num_regs = 2


model = mini_XCEPTION(input_shape, num_classes, num_regs)
losses = {"predictions_exp": "categorical_crossentropy","predictions_etc": "mean_squared_error",}
model.compile(optimizer='adam', loss=losses, metrics={'predictions_etc': 'mse', 'predictions_exp': 'accuracy'})
model.summary()

dataset_name = 'affectnet'

data_loader = DataManager(dataset_path='./datasets/AffectNet/', target_size=input_shape[:2],rescale=1.0 / 255.0)
train_img_path, train_exp, train_etc = data_loader.get_train_dataset()
val_images, val_exp, val_etc = data_loader.get_val_dataset()


log_file_path = base_path + dataset_name + '.log' # [ ]]] > 로그 파일 위치 ~
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
callbacks = [csv_logger, early_stop, reduce_lr] # [[[] > 콜백

data_generator = DataGenerator(train_img_path, train_exp, train_etc, batch_size, target_size=input_shape[:2], rescale=1.0 / 255.0)

model.fit(data_generator, steps_per_epoch=len(train_img_path) // batch_size,
          epochs=num_epochs, verbose=1, callbacks=callbacks,
          validation_data=(val_images, {'predictions_exp': val_exp, 'predictions_etc': val_etc}))

model.save(trained_models_path + '.h5')

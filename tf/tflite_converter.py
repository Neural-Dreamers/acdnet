import glob
import os

import tensorflow as tf

valid_path = False

while not valid_path:
    model_path = input("Enter model path\n:")
    file_paths = glob.glob(os.path.join(os.getcwd(), model_path))
    if len(file_paths) > 0 and os.path.isfile(file_paths[0]):
        model_path = file_paths[0]
        print('Model has been found at: {}'.format(model_path))
        valid_path = True

model_name = os.path.basename(model_path).split('.')[0]
tflite_model_save_dir = os.path.join(os.getcwd(), 'tf/tflite_models')

if not os.path.exists(tflite_model_save_dir):
    os.makedirs(tflite_model_save_dir)

model = tf.keras.models.load_model(model_path)
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_save_path = os.path.join(tflite_model_save_dir, model_name + 'lite' + ".tflite")

with open(tflite_model_save_path, 'wb') as f:
    f.write(tflite_model)

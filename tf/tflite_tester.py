import sys
import os

import numpy as np
import librosa
import tensorflow as tf


def padding(sound, size):
    diff = size - len(sound)
    return np.pad(sound, (diff//2, diff-(diff//2)), 'constant')


def multi_crop(sound, audio_length, n_crops):
    stride = (len(sound) - audio_length) // (n_crops - 1)
    sounds = [sound[stride * i: stride * i + audio_length] for i in range(n_crops)]
    return np.array(sounds)


model_path = None

while model_path is None:
    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    else:
        print('Enter model name as command line argument')
        exit()

print('Model has been found at: {}'.format(model_path))

data = os.path.join(os.getcwd(), 'datasets/fsc22/tflite_test_data')
raw_audio_files = [f for f in os.listdir(data)]
original_input_length = 100000

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_length = input_shape[2]
crops = 3

predictions = []

for raw_audio_file in raw_audio_files:
    filename = os.path.join(data, raw_audio_file)
    label = raw_audio_file.split('-')[-1].split('.')[0]

    normalised_audio, sample_rate = librosa.load(filename, sr=20000, mono=True)

    padded_audio = padding(normalised_audio, original_input_length)
    cropped_audios = multi_crop(padded_audio, input_length, crops)

    for audio in cropped_audios:
        np_audio = np.array(audio, dtype=np.float32).reshape((1, 1, input_length, 1))
        interpreter.set_tensor(input_details[0]['index'], np_audio)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = output_data.argmax() + 1
        predictions.append(pred)

print(predictions)

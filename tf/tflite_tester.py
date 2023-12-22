import csv
import os
import sys
import tracemalloc
import datetime

import numpy as np
import tflite_runtime.interpreter as tflite
from pydub import AudioSegment


def padding(sound, size):
    diff = size - len(sound)
    return np.pad(sound, (diff // 2, diff - (diff // 2)), 'constant')


def multi_crop(sound, audio_length, n_crops):
    stride = (len(sound) - audio_length) // (n_crops - 1)
    sounds = [sound[stride * i: stride * i + audio_length] for i in range(n_crops)]
    return np.array(sounds)


def inference():
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_length = input_shape[2]
    sr = 20000
    crops = 3

    y_pred = []
    y = []

    for raw_audio_file in raw_audio_files:
        filename = os.path.join(data, raw_audio_file)
        label = raw_audio_file.split('-')[-1].split('.')[0]

        # Read the WAV file using pydub and set the frame rate to 20,000 Hz
        audio = AudioSegment.from_file(filename, format="wav").set_frame_rate(sr).set_channels(1)

        # Get raw PCM data as a byte string
        raw_data = audio.raw_data

        # Convert the byte string to a NumPy array of 16-bit integers
        audio_frames = np.frombuffer(raw_data, dtype=np.int16)

        # Convert the integer audio data to float in the range [-1, 1]
        normalised_audio = audio_frames.astype(np.float32) / 32767.0

        if len(normalised_audio) > original_input_length:
            continue

        padded_audio = padding(normalised_audio, original_input_length)
        cropped_audios = multi_crop(padded_audio, input_length, crops)

        for audio in cropped_audios:
            np_audio = np.array(audio, dtype=np.float32).reshape((1, 1, input_length, 1))

            # start measuring inference time
            st = datetime.datetime.now()

            interpreter.set_tensor(input_details[0]['index'], np_audio)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = output_data.argmax() + 1

            # end measuring inference time
            et = datetime.datetime.now()
            infer_time = (et - st).total_seconds() * 1000

            infer_times.append(infer_time)

            y.append(label)
            y_pred.append(pred)

    return y_pred, y


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

infer_times = []

tracemalloc.start()

predictions, labels = inference()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

print('Current memory [MB]: {}, Peak memory [MB]: {}'.format(round(current / (1024 * 1024), 4),
                                                             round(peak / (1024 * 1024), 4)))

print(f"Infer time for {len(infer_times)} samples: {np.sum(infer_times)} ms")
print(f"Infer time per sample: {np.mean(infer_times)} ms")

# Combine arrays into a list of rows
rows = list(zip(labels, predictions))

tflite_model_pred_save_dir = os.path.join(os.getcwd(), 'tf/predictions')

if not os.path.exists(tflite_model_pred_save_dir):
    os.makedirs(tflite_model_pred_save_dir)

# Specify the file name
csv_file_name = os.path.join(tflite_model_pred_save_dir, 'output.csv')

# Open the CSV file in write mode
with open(csv_file_name, 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write header row
    csv_writer.writerow(['y', 'y_pred'])

    # Write the rows to the CSV file
    csv_writer.writerows(rows)

print(f'The arrays have been successfully written to {csv_file_name}.')

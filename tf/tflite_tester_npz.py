import math
import os
import sys

import numpy as np
import tensorflow as tf


class Tester:
    def __init__(self):
        self.testX = None
        self.testY = None

    # Loading Test data
    def load_data(self):
        data = np.load(os.path.join(os.getcwd(), 'datasets/fsc22/test_data_20khz/fold1_test3900.npz'),
                       allow_pickle=True)
        self.testX = data['x']
        self.testY = data['y']
        print(self.testX.shape)
        print(self.testY.shape)

    # Test the model with test data
    def validate(self, input_config, model, output_config):
        y_pred = None
        y_target = self.testY
        batch_size = (batchSize // nCrops) * nCrops
        audio_shape = input_config[0]['shape']

        for idx in range(math.ceil(len(self.testX) / batch_size)):
            x = self.testX[idx * batch_size: (idx + 1) * batch_size]
            scores = None

            for audio in x:
                audio = audio.reshape(audio_shape)
                model.set_tensor(input_config[0]['index'], audio)
                model.invoke()
                output_data = model.get_tensor(output_config[0]['index'])
                scores = output_data if scores is None else np.concatenate((scores, output_data))

            y_pred = scores if y_pred is None else np.concatenate((y_pred, scores))

        acc = self.compute_accuracy(y_pred, y_target)
        return acc

    # Calculating average prediction (10 crops) and final accuracy
    def compute_accuracy(self, y_pred, y_target):
        # Reshape y_pred to shape it like each sample contains 10 samples.
        y_pred = y_pred.reshape(y_pred.shape[0] // nCrops, nCrops, y_pred.shape[1])
        y_target = y_target.reshape(y_target.shape[0] // nCrops, nCrops, y_target.shape[1])

        # Calculate the average of class predictions for 10 crops of a sample
        y_pred = np.mean(y_pred, axis=1)
        y_target = np.mean(y_target, axis=1)

        # Get the indices that have the highest average value for each sample
        y_pred = y_pred.argmax(axis=1)
        y_target = y_target.argmax(axis=1)
        accuracy = (y_pred == y_target).mean() * 100

        return accuracy

    # Load and test the saved model
    def TestModel(self, model_file):
        interpreter = tf.lite.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()
        print('Model Loaded.')

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.load_data()
        print('Test dataset loaded.')
        val_acc = self.validate(input_details, interpreter, output_details)
        print('Testing - Acc(top1) {:.2f}%'.format(val_acc))


if __name__ == '__main__':
    sr = 20000
    inputLength = 30225
    nCrops = 10
    batchSize = 64

    model_path = None

    while model_path is None:
        if len(sys.argv) == 2:
            model_path = sys.argv[1]
        else:
            print('Enter model name as command line argument')
            exit()

    print('Model has been found at: {}'.format(model_path))

    tester = Tester()
    tester.TestModel(model_path)

import sys
import os
import glob

import torch
import onnx
import tensorflow as tf
import onnx_tf

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'common'))

import common.opts as opts
import resources.models as models
import resources.calculator as calc

opt = opts.parse()

opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'device: {opt.device}')

valid_path = False

while not valid_path:
    model_path = input("Enter model path\n:")
    file_paths = glob.glob(os.path.join(os.getcwd(), model_path))
    if len(file_paths) > 0 and os.path.isfile(file_paths[0]):
        opt.model_path = file_paths[0]
        opt.model_name = os.path.basename(opt.model_path).split(".")[0]
        print('Model has been found at: {}'.format(opt.model_path))
        valid_path = True

# Load the PyTorch ACDNAS model
curr_dir = os.getcwd()
net_path = opt.model_path

state = torch.load(net_path, map_location=opt.device)
config = state['config']
weight = state['weight']
net = models.GetACDNetModel(opt.inputLength, nclass=opt.nClasses[opt.dataset], sr=opt.sr, channel_config=config).to(opt.device)
net.load_state_dict(weight)

calc.summary(net, (1, 1, opt.inputLength))

onnx_model_dir = os.path.join(curr_dir, 'torch/torch-tflite_convert/onnx_models')
if not os.path.exists(onnx_model_dir):
    os.makedirs(onnx_model_dir)

# Export the PyTorch model to ONNX format
input_shape = (1, 1, 1, opt.inputLength)
dummy_input = torch.randn(input_shape)
onnx_model_path = os.path.join(onnx_model_dir, f'{opt.model_name}.onnx')
# torch.onnx.export(net, dummy_input, onnx_model_path, verbose=False)

torch.onnx.export(net, dummy_input, onnx_model_path,
                  input_names=['input'], output_names=['output'])

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

tf_model_dir = os.path.join(curr_dir, 'torch/torch-tflite_convert/tf_models')
if not os.path.exists(tf_model_dir):
    os.makedirs(tf_model_dir)

# Convert the ONNX model to TensorFlow format
tf_model_path = os.path.join(tf_model_dir, f'{opt.model_name}.h5')

tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

tflite_model_dir = os.path.join(curr_dir, 'torch/torch-tflite_convert/tflite_models')

if not os.path.exists(tflite_model_dir):
    os.makedirs(tflite_model_dir)

# Save the TensorFlow Lite model to a file
with open(os.path.join(tflite_model_dir, f'{opt.model_name}.tflite'), 'wb') as f:
    f.write(tflite_model)

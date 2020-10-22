

# https://www.tensorflow.org/lite/performance/post_training_integer_quant
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model

#import tensorflow.python.keras.backend as K
from tensorflow.keras import backend as K
#from tensorflow.keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.utils import CustomObjectScope

import numpy as np

from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from tensorflow.python.keras.utils.vis_utils import plot_model
from layers.AnchorBoxesLayer import AnchorBoxes


from tqdm import tqdm
import os
import cv2
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()



with CustomObjectScope({'relu6': tf.nn.relu6,'AnchorBoxes':AnchorBoxes}):
    converter = tf.lite.TFLiteConverter.from_keras_model_file('./ssd_models.h5')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter.convert()
    open("./ssd_models_fp16.tflite", "wb").write(tflite_fp16_model)

#fp32 17295924
#fp16 8689296


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./ssd_models_fp16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)





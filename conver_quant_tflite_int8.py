

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

tf.enable_eager_execution()

def rep_data_gen():
    input_shape = (1,300,300,3)
    img_dir = '/media/zhu/0003E52A000920B8/procedure/mobilenet_v2_ssdlite_keras-master/experiments/imgs/'
    BATCH_SIZE=2
    a = []
    img_list = os.listdir(img_dir)
    for file_name in tqdm(img_list):
        img = cv2.imread(img_dir + file_name)
        img = cv2.resize(img, (input_shape[1],input_shape[2]))
        img = img.astype(np.float32)
        img = img / 255.0
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
        a.append(img)
    a = np.array(a)
    print(a.shape) # a is np array of 160 3D images
    for input_value in tf.data.Dataset.from_tensor_slices(a).batch(1).take(100):
        yield [input_value]





with CustomObjectScope({'relu6': tf.nn.relu6,'AnchorBoxes':AnchorBoxes}):
    converter = tf.lite.TFLiteConverter.from_keras_model_file('./ssd_models.h5')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.allow_custom_ops=True
    converter.representative_dataset = rep_data_gen
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    open("./ssd_model_int8.tflite", "wb").write(tflite_model_quant)





# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./ssd_model_int8.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)





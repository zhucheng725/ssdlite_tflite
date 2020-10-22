

#####load model

import keras
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.utils.generic_utils import CustomObjectScope

import numpy as np




with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    base_model = load_model('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/finetuning_mbv2_segnet.h5')

base_model.summary()

output_feat = base_model.get_layer('bbn_stage1_block1_bn').output
input_feat = base_model.input
new_model = keras.models.Model(input_feat, output_feat)

new_model.summary()
new_model.save('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/new_model.h5')

#####load model

import keras
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.utils.generic_utils import CustomObjectScope

import numpy as np
import cv2


with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    base_model = load_model('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/new_model.h5')

img =  cv2.imread('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/test.jpg')
img = img.astype(np.float32)
img[:, :, 0] -= 103.939
img[:, :, 1] -= 116.779
img[:, :, 2] -= 123.68
img = img[:, :, ::-1]
img = img.reshape((1,300,300,3))
a = base_model.predict(img)
a = a.reshape((150,150,32))

a[:,:,0]









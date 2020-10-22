
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



# model config
batch_size = 16
image_size = (300, 300, 3)
n_classes = 80
mode = 'training'
l2_regularization = 0.0005
min_scale = 0.1
max_scale = 0.9
scales = None
aspect_ratios_global = None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = None
offsets = None
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.01
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False

K.clear_session()

model = mobilenet_v2_ssd(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes)

weights_path = '/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/pretrained_weights/ssdlite_coco_loss-4.8205_val_loss-4.1873.h5'
model.load_weights(weights_path, by_name=True)

model.summary()
#plot_model(model,to_file='/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/model.png', show_shapes= True)

cut_model  = keras.models.Model(model.input, model.get_layer('bbn_stage3_block3_add').output)

cut_model.save('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/ssd_mbv2_no_top.h5')




#####load model

import keras
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.utils.generic_utils import CustomObjectScope

import numpy as np



with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    base_model = load_model('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/ssd_mbv2_no_top.h5')

base_model.summary()



def segnet_decoder(f, n_classes, n_up=3):
    assert n_up >= 2
    o = f
    o = (keras.layers.ZeroPadding2D((1, 1)))(o)
    o = (keras.layers.Conv2D(512, (3, 3), padding='valid'))(o)
    o = (keras.layers.BatchNormalization())(o)
    o = (keras.layers.UpSampling2D((2, 2)))(o)
    o = (keras.layers.ZeroPadding2D((1, 1)))(o)
    o = (keras.layers.Conv2D(256, (3, 3), padding='valid'))(o)
    o = (keras.layers.BatchNormalization())(o)
    for _ in range(n_up-2):
        o = (keras.layers.UpSampling2D((2, 2)))(o)
        o = (keras.layers.ZeroPadding2D((1, 1)))(o)
        o = (keras.layers.Conv2D(128, (3, 3), padding='valid'))(o)
        o = (keras.layers.BatchNormalization())(o)
    o = (keras.layers.UpSampling2D((2, 2)))(o)
    #o = (keras.layers.ZeroPadding2D((1, 1)))(o)
    o = (keras.layers.Conv2D(64, (3, 3), padding='valid'))(o)
    o = (keras.layers.BatchNormalization())(o)
    o = keras.layers.Conv2D(n_classes, (3, 3), padding='valid')(o)
    return o



def my_finetune_layers(input_feat, output_feat, n_classes = 5):
    x = output_feat
    o = segnet_decoder(x, n_classes, n_up=3)
    o = (keras.layers.Reshape((300*300,-1), name = 'finetuning_reshape_1'))(o)
    o = keras.layers.Activation('softmax', name = 'finetuning_softmax')(o)
    #x = keras.layers.convolutional.AveragePooling2D((7,7), name = 'new_AveragePooling2D')(x)
    #x = keras.layers.core.Flatten()(x)
    #x = keras.layers.Dense(10, activation='softmax')(x)
    new_model = keras.models.Model(input_feat, o)
    return new_model


#input_img = keras.layers.Input(shape=(300,300,3), name = 'new_input')

output_feat = base_model.output
input_feat = base_model.input

new_model = my_finetune_layers(input_feat, output_feat, n_classes = 5)

new_model.summary()
new_model.save('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/new_model.h5')











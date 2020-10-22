
import keras
from keras.models import load_model
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.utils.generic_utils import CustomObjectScope

import numpy as np
import cv2


VOC_COLOR = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]], dtype=np.uint8)

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    model = load_model('/media/kirito/1T/procedure/onnx/mobilenet_v2_ssdlite_keras-master/finetuning_mbv2_segnet.h5')

model.summary()



for k in range(10):
    img =  cv2.imread('/media/kirito/1T/procedure/lane_detect/wallmarket/cocopark/1/image_' + str(k)+'.jpg')
    img =  cv2.resize(img,(300,300), interpolation =  cv2.INTER_AREA)
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    img = img.reshape((1,300,300,3))
    a = model.predict(img)
    a = a.reshape((300,300,5))
    pre = np.zeros((300,300,3))
    for i in range(300):
        for j in range(300):
            result_color = np.argmax(a[i,j,:])
            #if result_color == 7 or result_color == 15:
            pre[i,j] =  VOC_COLOR[result_color]
    img =  cv2.imread('/media/kirito/1T/procedure/lane_detect/wallmarket/cocopark/1/image_' + str(k)+'.jpg')
    pre =  cv2.resize(pre,(1280,720), interpolation =  cv2.INTER_AREA)
    pre = pre.astype(np.float32)
    img = img.astype(np.float32)
    overlapping = cv2.addWeighted(img, 0.7, pre, 1.0, 0)
    cv2.imwrite('/media/kirito/1T/procedure/onnx/mobilenet_segnet/all/output/cocopark/image_'+ str(k)+'.jpg',overlapping)
    print(k)






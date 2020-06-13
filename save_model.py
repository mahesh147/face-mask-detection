import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, decode

import numpy as np
import tensorflow as tf

flags.DEFINE_string('weights', None, 'name of weights file')
flags.DEFINE_integer('size', 416, 'input size to resize the image')
#flag.DEFINE_boolean('debug', True, 'print debug info')


def main(argv):

    weights = FLAGS.weights 
    input_size = FLAGS.size

    NUM_CLASS = 2

    print(f'[DEBUG][save_model] Path to weights : weights/{FLAGS.weights}')
    print(f'[DEBUG][save_model] Size : {FLAGS.size}')

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    print(f'[INFO][save_model] Created input_layer of size {input_size}')
    print(f'[DEBUG][save_model] input_layer : {input_layer}')

    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    print(f'[DEBUG][save_model] feature_maps : {feature_maps}')
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensors.append(decode(fm, NUM_CLASS, i))

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, 'weights/' + FLAGS.weights)

    print(f'[INFO][save_model] Saving model... ')

    model.save(f'models/{weights.split(".")[0]}-size-{input_size}.h5')
    
    print(f'[INFO][save_model] Model saved to models/{weights.split(".")[0]}-size-{input_size}.h5')


if __name__ == '__main__':
    app.run(main)

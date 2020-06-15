import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, decode
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('weights', None, 'name of weights file')
flags.DEFINE_string('image_path', None, 'path to the image')
flags.DEFINE_float('score_thresh', 0.25, 'prediction score threshold')
flags.DEFINE_float('iou_thresh', 0.213, 'iou prediction score threshold')
flags.DEFINE_integer('size', 416, 'input size to resize the image')
flags.DEFINE_string('save_path', None, 'path to save the image')

def main(argv):
    NUM_CLASS = 2
    ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75,
               76, 55, 72, 146, 142, 110,
               192, 243, 459, 401]
    ANCHORS = np.array(ANCHORS, dtype=np.float32)
    ANCHORS = ANCHORS.reshape(3, 3, 2)
    STRIDES = [8, 16, 32]
    XYSCALE = [1.2, 1.1, 1.05]
    input_size = FLAGS.size
    image_path = FLAGS.image_path
    score_thresh = FLAGS.score_thresh
    iou_thresh = FLAGS.iou_thresh
    save_path = FLAGS.save_path

    print(f'[DEBUG][image] input_size : {input_size}')
    print(f'[DEBUG][image] image_path : {image_path}')
    print(f'[DEBUG][image] score_thresh : {score_thresh}')
    print(f'[DEBUG][image] iou_thresh : {iou_thresh}')

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    print(f'[DEBUG][image] original_image_size : {original_image_size}')

    image_data = utils.image_preprocess(
        np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    print('[INFO] Bulding Yolov4 architecture')

    tic = time.perf_counter()
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    print(f'[INFO][image] Created input_layer of size {input_size}')
    print(f'[DEBUG][image] input_layer : {input_layer}')

    feature_maps = YOLOv4(input_layer, NUM_CLASS)

    print(f'[DEBUG][image] feature_maps : {feature_maps}')

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensors.append(decode(fm, NUM_CLASS, i))

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, FLAGS.weights)

    toc = time.perf_counter()
    print(f'[INFO] Architecture built.')
    print(f'[DEBUG][image] Execution took {(1000 * (toc - tic)):0.4f} ms')

    pred_bbox = model.predict(image_data)

    print(f'[INFO][image] Finished initial predication on image')

    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

    bboxes = utils.postprocess_boxes(
        pred_bbox, original_image_size, input_size, score_thresh)

    bboxes = utils.nms(bboxes, iou_thresh, method='nms')

    image = utils.draw_bbox(original_image, bboxes)

    image = Image.fromarray(image)

    image.show()

    if (save_path):
        image.save(save_path)
        print(f'[INFO][image] Detected image saved to {save_path}')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

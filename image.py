import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('model', None, 'path to the model')
flags.DEFINE_string('image_path', None, 'path to the image')
flags.DEFINE_float('score_thresh', 0.25, 'prediction score threshold')
flags.DEFINE_float('iou_thresh', 0.213, 'iou prediction score threshold')

def main(argv):
    NUM_CLASS = 2
    ANCHORS = [12,16, 19,36, 40,28, 36,75, 
                76,55, 72,146, 142,110, 
                192,243, 459,401]
    ANCHORS = np.array(ANCHORS, dtype=np.float32)
    ANCHORS = ANCHORS.reshape(3, 3, 2)
    STRIDES = [8, 16, 32]
    XYSCALE = [1.2, 1.1, 1.05] 
    model = FLAGS.model
    input_size = int(model.split(".")[0].split("size")[1][1:])
    image_path = FLAGS.image_path
    score_thresh = FLAGS.score_thresh
    iou_thresh = FLAGS.iou_thresh

    print(f'[DEBUG][image] input_size : {input_size}')
    print(f'[DEBUG][image] image_path : {image_path}')
    print(f'[DEBUG][image] score_thresh : {score_thresh}')
    print(f'[DEBUG][image] iou_thresh : {iou_thresh}')

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    print(f'[DEBUG][image] original_image_size : {original_image_size}')

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    print(f'[INFO][image] Loading {FLAGS.model}')

    tic = time.perf_counter()
    model = tf.keras.models.load_model(model, custom_objects={'tf': tf})
    toc = time.perf_counter()
    print('[INFO][image] Model loaded.')
    print(f'[DEBUG][image] Execution took {toc - tic:0.4f} seconds')

    pred_bbox = model.predict(image_data)

    print(f'[INFO][image] Finished initial predication on image')

    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, score_thresh)

    bboxes = utils.nms(bboxes, iou_thresh, method='nms')

    image = utils.draw_bbox(original_image, bboxes)

    image = Image.fromarray(image)

    image.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

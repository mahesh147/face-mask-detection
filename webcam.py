import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('model', None, 'path to the model')
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
    score_thresh = FLAGS.score_thresh
    iou_thresh = FLAGS.iou_thresh


    print(f'[DEBUG][image] input_size : {input_size}')
    print(f'[DEBUG][image] score_thresh : {score_thresh}')
    print(f'[DEBUG][image] iou_thresh : {iou_thresh}')

    print(f'[INFO][image] Loading {model}')

    tic = time.perf_counter()
    model = tf.keras.models.load_model(model, custom_objects={'tf': tf})
    toc = time.perf_counter()
    print('[INFO][image] Model loaded.')
    print(f'[DEBUG][image] Execution took {toc - tic:0.4f} seconds')

    vid = cv2.VideoCapture(0)

    while True:
        return_value, frame = vid.read()
        if return_value:
            print(f'[DEBUG] Got video capture')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image! Try with another video format")
        frame_size = frame.shape[:2]
        
        image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        pred_bbox = model.predict(image_data)
        print(f'[INFO][image] Finished initial predication on image')

        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, score_thresh)

        bboxes = utils.nms(bboxes, iou_thresh, method='nms')

        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)

        print(info)

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        print(result.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    vid.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



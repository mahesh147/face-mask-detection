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
flags.DEFINE_float('score_thresh', 0.25, 'prediction score threshold')
flags.DEFINE_float('iou_thresh', 0.213, 'iou prediction score threshold')
flags.DEFINE_integer('size', 416, 'input size to resize the image')
flags.DEFINE_string('save_path', None, 'path to save the video')

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
    score_thresh = FLAGS.score_thresh
    iou_thresh = FLAGS.iou_thresh
    save_path = FLAGS.save_path

    print(f'[DEBUG][webcam] input_size : {input_size}')
    print(f'[DEBUG][webcam] score_thresh : {score_thresh}')
    print(f'[DEBUG][webcam] iou_thresh : {iou_thresh}')

    print('[INFO] Bulding Yolov4 architecture')
    tic = time.perf_counter()

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    print(f'[INFO][webcam] Created input_layer of size {input_size}')
    print(f'[DEBUG][webcam] input_layer : {input_layer}')

    feature_maps = YOLOv4(input_layer, NUM_CLASS)

    print(f'[DEBUG][webcam] feature_maps : {feature_maps}')
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensors.append(decode(fm, NUM_CLASS, i))

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, FLAGS.weights)

    toc = time.perf_counter()
    print(f'[INFO] Architecture built.')
    print(f'[DEBUG][webcam] Execution took {(1000 * (toc - tic)):0.4f} ms')

    vid = cv2.VideoCapture(0)

    if save_path:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        print(f"[DEBUG][video] Video CODEC : {FLAGS.save_path.split('.')[1]}")
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(FLAGS.save_path, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            print(f'[DEBUG] Got video capture')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image! Try with another video format")
        frame_size = frame.shape[:2]

        image_data = utils.image_preprocess(
            np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.perf_counter()

        pred_bbox = model.predict(image_data)
        print(f'[INFO][webcam] Finished initial predication on image')

        pred_bbox = utils.postprocess_bbbox(
            pred_bbox, ANCHORS, STRIDES, XYSCALE)

        bboxes = utils.postprocess_boxes(
            pred_bbox, frame_size, input_size, score_thresh)

        bboxes = utils.nms(bboxes, iou_thresh, method='nms')

        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.perf_counter()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "fdpms: %.2f ms" % (1000*exec_time)

        print(info)

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        print(result.shape)
        if save_path:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    out.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

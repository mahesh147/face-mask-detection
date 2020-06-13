##  Changelog

All notable changes to this project will be documented in this file.

### [0.1.0]  - 12-06-2020

### Removed 

- Removed the android/, mAP/ and scripts/ directories.
- Removed files related to training and conversion of weights.
- Removed the .png files from data/ directory.
- Removed the classes/ and dataset/ directories from data/.
- Removed dataset.py from core/ directory
- Removed the following functions in core/utils.py:
  - load_weights_tiny
  - load_weights_v3
- Removed the following functions in core/yolov4.py:
  - YOLOv3
  - YOLOv3_tiny
  - decode_train
- Removed the following functions in core/backbone.py
  - darknet53
  - darknet53_tiny

### Added

- save_model.py to save each darknet weights to keras compatible .h5 weights 
- Added [INFO] and [DEBUG] log messages
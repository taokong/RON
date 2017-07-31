#!/usr/bin/env bash
python tools/demo_video.py \
    --model models/pascalvoc/VGG16/test320cudnn.prototxt \
    --weights data/RON_models/RON320_VOC0712_VOC07.caffemodel


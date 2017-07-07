#!/usr/bin/env bash
python ./tools/test_net.py --gpu 0 \
  --def models/pascalvoc/VGG16/test320cudnn.prototxt \
  --net data/RON_models/RON320_VOC0712_VOC07.caffemodel \
  --imdb voc_2007_test

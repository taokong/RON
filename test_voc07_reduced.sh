#!/usr/bin/env bash
python ./tools/test_net.py --gpu 0 \
  --def models/pascalvoc/VGG16-REDUCED/test320cudnn.prototxt \
  --net output/default/voc_2007_trainval+voc_2012_trainval/RON-REDUCED_320_iter_1000.caffemodel \
  --imdb voc_2007_test

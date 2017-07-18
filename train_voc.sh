#!/usr/bin/env bash
python tools/train_net.py --gpu 0 \
  --solver models/pascalvoc/VGG16/solver.prototxt \
  --imdb voc_2007_trainval+voc_2012_trainval \
  --weights data/ImageNet_models/VGG16_layers_fully_conv.caffemodel \
  --batchsize 20 \
  --iters 120000

#!/usr/bin/env bash
python tools/train_net.py --gpu 0 \
  --solver models/pascalvoc/VGG16-REDUCED/solver.prototxt \
  --imdb voc_2007_trainval+voc_2012_trainval \
  --weights data/ImageNet_models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
  --batchsize 30 \
  --iters 120000

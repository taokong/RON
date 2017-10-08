#!/usr/bin/env bash
python tools/train_net.py --gpu 0 \
  --solver models/ai_human/mobilenet/solver.prototxt \
  --imdb ai_20170911_validation \
  --weights data/ImageNet_models/mobilenet.caffemodel \
  --batchsize 2 \
  --iters 120000

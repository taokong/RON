# --------------------------------------------------------
# RON
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Tao Kong
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, unmap
import matplotlib.pyplot as plt
DEBUG = False

class OHEMRpnLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        height, width = bottom[0].data.shape[-2:]
        layer_params = yaml.load(self.param_str)
        self._sample_num = layer_params['sample_num']
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        self._ndim = cfg.TRAIN.IMS_PER_BATCH

        # labels
        top[0].reshape(self._ndim, 1, height, width)
        

    def forward(self, bottom, top):

        # map of shape (..., H, W)
        all_labels_up = bottom[1].data


        all_labels = all_labels_up.flatten()

        background_inds = np.where(all_labels == 0)[0]

        gt_num = len(np.where(all_labels > 0)[0])

        sample_num = gt_num * self._sample_num

        if(sample_num < len(background_inds)):
            all_losses_up = bottom[0].data
            all_losses = all_losses_up.flatten()[background_inds]

            loss_flag_value = np.sort(-all_losses)[sample_num] * -1

            ignore_inds = np.where((all_losses_up < loss_flag_value) & (all_labels_up == 0))

            all_labels_up[ignore_inds] = -1

        if DEBUG:
            print 'ohem: num_positive', np.sum(all_labels_up > 0)
            print 'ohem: num_negative', np.sum(all_labels_up == 0)

        top[0].reshape(*all_labels_up.shape)
        top[0].data[...] = all_labels_up


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

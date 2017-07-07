# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Tao Kong, 2016-11-22
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
import matplotlib.pyplot as plt

class DetLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes.
    """

    def setup(self, bottom, top):
        """
        bottom[0]: rpn score map
        bottom[1]: det score map
        bottom[2]: bbox_pred_delta    
        top[0]: ROIs
        top[1]: scores
        """
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        params = layer_params['stride_scale_numcls']

        feat_stride_str, scale_str, cls_str = params.split(',')
        self._feat_stride = int(feat_stride_str)
        anchor_scale = int(scale_str)
        self._numclasses = int(cls_str)

        anchors = generate_anchors(base_size = cfg.MINANCHOR, scales=np.array([anchor_scale, anchor_scale + 1]), ratios = [0.333, 0.5, 1, 2, 3])
        self._num_anchors = anchors.shape[0]

        # 1. Generate proposals from bbox deltas and shifted anchors
        _ndim, _tmp, self._height, self._width = bottom[0].data.shape
        self._ndim = cfg.TEST.BATCH_SIZE

        # Enumerate all shifts
        shift_x = np.arange(0, self._width) * self._feat_stride
        shift_y = np.arange(0, self._height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        K = shifts.shape[0]
        self._anchors = anchors.reshape((1, self._num_anchors, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        self._anchors = self._anchors.reshape((K * self._num_anchors, 4))

        top[0].reshape(self._ndim, 1, 4)
        top[1].reshape(self._ndim, 1, self._numclasses)

    def forward(self, bottom, top):

        # get all of the mini-batches
        all_scores_rpn = bottom[0].data[:, self._num_anchors:, :, :]
        all_scores_rpn = all_scores_rpn.transpose((0, 2, 3, 1)).reshape((-1, 1))

        all_scores_det = bottom[1].data[:, 1:, :, :].reshape(self._ndim, self._numclasses - 1, self._num_anchors, self._height, self._width)
        all_scores_det = all_scores_det.transpose((0, 3, 4, 2, 1)).reshape((-1, self._numclasses - 1))

        all_scores = np.hstack((all_scores_rpn, all_scores_det)).reshape((self._ndim, -1, self._numclasses))

        all_bbox_deltas_rpn =  bottom[2].data.transpose((0, 2, 3, 1)).reshape((-1, 4))
        all_anchors = np.tile(self._anchors, (self._ndim, 1))
        all_proposals = bbox_transform_inv(all_anchors, all_bbox_deltas_rpn)
        all_proposals = all_proposals.reshape((self._ndim, -1, 4))

        top[0].reshape(*(all_proposals.shape))
        top[0].data[...] = all_proposals
        top[1].reshape(*(all_scores.shape))
        top[1].data[...] = all_scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


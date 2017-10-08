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

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        #anchor_scales = [layer_params['scale']]

        params = layer_params['stride_scale_border_batchsize']

        feat_stride_str, scale_str, border_str, batch_str = params.split(',')
        self._feat_stride = float(feat_stride_str)
        self._scales = float(scale_str)       
        self._allowed_border = float(border_str)
        self._batch = float(batch_str)
        self._anchors = generate_anchors(base_size = cfg.MINANCHOR, scales=np.array([2, 4, 8]), ratios = [0.5, 1, 2])
        self._num_anchors = self._anchors.shape[0]


        self._height, self._width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', self._height, 'width', self._width

        self._ndim = cfg.TRAIN.IMS_PER_BATCH

        shift_x = np.arange(0, self._width) * self._feat_stride
        shift_y = np.arange(0, self._height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()

        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, self._num_anchors, 4)) +
                shifts.reshape((1, K, 4)).transpose((1, 0, 2)))

        self._all_anchors = all_anchors.reshape((K * self._num_anchors, 4))
        self._total_anchors = int(K * self._num_anchors)

        # only keep anchors inside the image
        self._inds_inside = np.where(
            (self._all_anchors[:, 0] >= -self._allowed_border) &
            (self._all_anchors[:, 1] >= -self._allowed_border) &
            (self._all_anchors[:, 2] < 320 + self._allowed_border) &  # width
            (self._all_anchors[:, 3] < 320 + self._allowed_border)    # height
        )[0]

        # labels
        top[0].reshape(self._ndim, 1, self._num_anchors * self._height, self._width)
        # bbox_targets
        top[1].reshape(self._ndim, self._num_anchors * 4, self._height, self._width)
        # bbox_inside_weights
        top[2].reshape(self._ndim, self._num_anchors * 4, self._height, self._width)
        # bbox_outside_weights
        top[3].reshape(self._ndim, self._num_anchors * 4, self._height, self._width)


    def forward(self, bottom, top):

        # map of shape (..., H, W)
        # GT boxes (x1, y1, x2, y2, label,dim)
        gt_boxes = bottom[1].data[:, 0:5]
        dim_inds = bottom[1].data[:, -1]



        if DEBUG:
            print 'total_anchors', self._total_anchors
            print 'inds_inside', len(self._inds_inside)

        # keep only inside anchors
        anchors = self._all_anchors[self._inds_inside, :]

        # generating overlaps of all anchors and gt_boxes
        batch_overlaps = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(gt_boxes, dtype=np.float))

        batch_gt_argmax_overlaps = batch_overlaps.argmax(axis=0)
        #generate for each dim
        all_labels = np.zeros((len(self._inds_inside) * self._ndim, ), dtype=np.float32)

        for i_dim in xrange(self._ndim):
            labels = np.ones((len(self._inds_inside), ), dtype=np.float32)  * -1

            inds_i = np.where(dim_inds == i_dim)[0]
            overlaps = batch_overlaps[:, inds_i]
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(self._inds_inside)), argmax_overlaps]

            gt_argmax_overlaps = batch_gt_argmax_overlaps[inds_i]

            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            labels_gt_inds = np.where(gt_max_overlaps >  cfg.TRAIN.BG_THRESH_HI - 0.2)[0]

            box_max = gt_argmax_overlaps[labels_gt_inds]
            labels[(max_overlaps <  cfg.TRAIN.BG_THRESH_HI)&(max_overlaps > cfg.TRAIN.BG_THRESH_LO)] = 0

            # fg label: for each gt, anchor with highest overlap when gt_argmax_overlaps > thresh
            labels[box_max] = 1
            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.FG_THRESH] = 1

            all_labels[i_dim * len(self._inds_inside): (i_dim + 1) * len(self._inds_inside)] = labels

        fg_inds = np.where(all_labels == 1)[0]

        if len(fg_inds) > 0:
            num_bg = len(fg_inds) *  (1.0 - cfg.TRAIN.FG_FRACTION) / (cfg.TRAIN.FG_FRACTION)
        else:
            num_bg = self._batch


        bg_inds = np.where(all_labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=int(len(bg_inds) - num_bg), replace=False)
            all_labels[disable_inds] = -1


        all_labels_top = np.zeros((self._ndim, 1, self._num_anchors * self._height, self._width), dtype = np.float32)

        all_bbox_targets_top = np.zeros((self._ndim, self._num_anchors*4, self._height, self._width), dtype = np.float32)
        all_bbox_inside_weights_top = np.zeros((self._ndim, self._num_anchors*4, self._height, self._width), dtype = np.float32)
        all_bbox_outside_weights_top = np.zeros((self._ndim, self._num_anchors*4, self._height, self._width), dtype = np.float32)


        for i_dim in xrange(self._ndim):
            labels = all_labels[i_dim * len(self._inds_inside): (i_dim + 1) * len(self._inds_inside)]
            inds_i = np.where(dim_inds == i_dim)[0]
            gt_boxes_i = gt_boxes[inds_i, :]

            overlaps = batch_overlaps[:, inds_i]
            argmax_overlaps = overlaps.argmax(axis=1)

            bbox_targets = np.zeros((len(self._inds_inside), 4), dtype=np.float32)
            bbox_inside_weights = np.zeros((len(self._inds_inside), 4), dtype=np.float32)
            bbox_outside_weights = np.zeros((len(self._inds_inside), 4), dtype=np.float32)

            bbox_targets = _compute_targets(anchors, gt_boxes_i[argmax_overlaps, :])
            bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels > 0)
            if num_examples < 1:
               num_examples = 1
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples

            bbox_outside_weights[labels == 1, :] = positive_weights
            bbox_outside_weights[labels == 0, :] = negative_weights

             # labels
            labels = unmap(labels, self._total_anchors, self._inds_inside, fill= -1)
            labels = labels.reshape((1, self._height, self._width, self._num_anchors)).transpose(0, 3, 1, 2)
            labels = labels.reshape((1, 1, self._num_anchors * self._height, self._width))

            bbox_targets = unmap(bbox_targets, self._total_anchors, self._inds_inside, fill=0)
            bbox_inside_weights = unmap(bbox_inside_weights, self._total_anchors, self._inds_inside, fill=0)
            bbox_outside_weights = unmap(bbox_outside_weights, self._total_anchors, self._inds_inside, fill=0)

            bbox_targets = bbox_targets.reshape((1, self._height, self._width, self._num_anchors * 4)).transpose(0, 3, 1, 2)
            bbox_inside_weights = bbox_inside_weights.reshape((1, self._height, self._width, self._num_anchors * 4)).transpose(0, 3, 1, 2)
            bbox_outside_weights = bbox_outside_weights.reshape((1, self._height, self._width, self._num_anchors * 4)).transpose(0, 3, 1, 2)

            all_labels_top[i_dim] = labels
            all_bbox_targets_top[i_dim] = bbox_targets
            all_bbox_inside_weights_top[i_dim] = bbox_inside_weights
            all_bbox_outside_weights_top[i_dim] = bbox_outside_weights

        top[0].reshape(*all_labels_top.shape)
        top[0].data[...] = all_labels_top

        top[1].reshape(*all_bbox_targets_top.shape)
        top[1].data[...] = all_bbox_targets_top

        top[2].reshape(*all_bbox_inside_weights_top.shape)
        top[2].data[...] = all_bbox_inside_weights_top

        top[3].reshape(*all_bbox_outside_weights_top.shape)
        top[3].data[...] = all_bbox_outside_weights_top



    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

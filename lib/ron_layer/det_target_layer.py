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
from fast_rcnn.bbox_transform  import unmap
import matplotlib.pyplot as plt

DEBUG = False

class DetTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        params = layer_params['stride_scale_border_batchsize_numcls']

        feat_stride_str, scale_str, border_str, batch_str, num_cls = params.split(',')
        self._feat_stride = float(feat_stride_str)
        self._scales = float(scale_str)       
        self._num_classes = int(num_cls)
        self._score_thresh = cfg.TRAIN.PROB  
        self._allowed_border = float(border_str)
        self._batch = float(batch_str)
              
        self._anchors = generate_anchors(base_size = cfg.MINANCHOR, scales=np.array([self._scales, self._scales + 1]), ratios = [0.333, 0.5, 1, 2, 3])
        self._num_anchors = self._anchors.shape[0]  
        
        height, width = bottom[0].data.shape[-2:]
        self._ndim = cfg.TRAIN.IMS_PER_BATCH

        # labels
        top[0].reshape(self._ndim, 1, self._num_anchors * height, width)

    def forward(self, bottom, top):

        scores = bottom[0].data[:,self._num_anchors:, :, :]

        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data[:, 0:5]
        # im_info
        im_info = bottom[2].data[0]
        dim_inds = bottom[1].data[:, -1]

        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, self._num_anchors, 4)) + 
                shifts.reshape((1, K, 4)).transpose((1, 0, 2)))

        all_anchors = all_anchors.reshape((K * self._num_anchors, 4))

        total_anchors = int(K * self._num_anchors)
        
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]
       
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)
        # keep only inside anchors

        anchors = all_anchors[inds_inside, :]

        if DEBUG:
            print 'anchors.shape', anchors.shape

        batch_overlaps = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(gt_boxes, dtype=np.float))

        batch_gt_argmax_overlaps = batch_overlaps.argmax(axis=0)

        #generate for each dim
        all_labels_ndim = np.zeros((len(inds_inside) * self._ndim, ), dtype=np.float32)   
        for i_dim in xrange(self._ndim):
            labels = np.ones((len(inds_inside), ), dtype=np.float32)  * -1     
            inds_i = np.where(dim_inds == i_dim)[0]
            gt_boxes_i = gt_boxes[inds_i, :]

            overlaps = batch_overlaps[:, inds_i]

            argmax_overlaps = overlaps.argmax(axis=1)

            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            
            gt_argmax_overlaps = batch_gt_argmax_overlaps[inds_i]
    
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

            labels_gt_inds = np.where(gt_max_overlaps > cfg.TRAIN.BG_THRESH_HI)[0]
            labels_gt = gt_boxes_i[labels_gt_inds, 4]
            gt_max_overlaps = gt_max_overlaps[labels_gt_inds]

            max_overlaps_argsort = np.argsort(-max_overlaps)
            box_max = gt_argmax_overlaps[labels_gt_inds]

            labels[max_overlaps < cfg.TRAIN.BG_THRESH_HI] = 0

            if len(box_max) > 0:
                labels[box_max] = labels_gt
   
            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.DET_POSITIVE_OVERLAP] = gt_boxes_i[argmax_overlaps[max_overlaps >= cfg.TRAIN.DET_POSITIVE_OVERLAP], -1]

            # filter boxes according to objectness scores
            labels_up = unmap(labels, total_anchors, inds_inside, fill= -1)
            labels_up = labels_up.reshape((height, width, self._num_anchors)).transpose(2, 0, 1)
            scores_i = scores[i_dim]
            ignore_inds = np.where(scores_i < self._score_thresh)
            labels_up[ignore_inds] = -1
            labels_up = labels_up.transpose(1, 2, 0).reshape((-1,1)).flatten()
            labels = labels_up[inds_inside]

            all_labels_ndim[i_dim * len(inds_inside): (i_dim + 1) * len(inds_inside)] = labels.copy()

        fg_inds = np.where(all_labels_ndim > 0)[0]
        if len(fg_inds) > 0:
            num_bg = len(fg_inds) *  (1.0 - cfg.TRAIN.FG_FRACTION) / (cfg.TRAIN.FG_FRACTION)
        else:
            num_bg = self._batch
     
        bg_inds = np.where(all_labels_ndim == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=int(len(bg_inds) - num_bg), replace=False)
            all_labels_ndim[disable_inds] = -1

        all_labels_top = np.zeros((self._ndim, 1, self._num_anchors * height, width), dtype = np.float32)
        all_bbox_targets_top = np.zeros((self._ndim, self._num_anchors*4, height, width), dtype = np.float32)
        all_bbox_inside_weights_top = np.zeros((self._ndim, self._num_anchors*4, height, width), dtype = np.float32)
        all_bbox_outside_weights_top = np.zeros((self._ndim, self._num_anchors*4, height, width), dtype = np.float32)


        for i_dim in xrange(self._ndim):
            labels = all_labels_ndim[i_dim * len(inds_inside): (i_dim + 1) * len(inds_inside)]

            labels = unmap(labels, total_anchors, inds_inside, fill= -1)
            # labels
            labels = labels.reshape((1, height, width, self._num_anchors)).transpose(0, 3, 1, 2) 
            labels = labels.reshape((1, 1, self._num_anchors * height, width))

            all_labels_top[i_dim] = labels
        
        if DEBUG:
            print 'det: num_positive', np.sum(all_labels_top > 0)
            print 'det: num_negative', np.sum(all_labels_top == 0)        

        top[0].reshape(*all_labels_top.shape)
        top[0].data[...] = all_labels_top
        

   

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass





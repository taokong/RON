# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Kong Tao
# date Dec. 03, 2016
# --------------------------------------------------------

"""Compute minibatch blobs for training RON network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes
from utils.blob import prep_im_for_blob, im_list_to_blob
from utils.cython_bbox import bbox_overlaps_gt
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
import copy

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch gt boxes from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=1)
    # Get the input image blob, formatted for caffe
    im_blob, im_scales, im_infos = _get_image_blob_augment(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 6), dtype=np.float32)
    im_info_blob = np.zeros((0, 2), dtype=np.float32)

    all_overlaps = []
    for im_i in xrange(num_images):
        labels = roidb[im_i]['gt_classes']
        im_rois = roidb[im_i]['boxes'].astype(np.float32)
            
        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i,:], random_scale_inds[0])
        labels_blob = np.zeros((len(labels),2), dtype = np.float32)
        labels_blob[:,0] = labels 
        labels_blob[:,1] = im_i     

        rois_blob_this_image = np.hstack((rois, labels_blob))
        
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        im_info_blob = np.vstack((im_info_blob, im_infos[im_i]))
        
    # _vis_minibatch(im_blob, rois_blob, all_overlaps)

    blobs = {'data': im_blob,
             'im_info': im_info_blob,
             'gt_rois': rois_blob}

    return blobs

def _get_rois(roidb):
    """Get the gt RoIs."""
    labels = roidb['gt_classes']
    rois = roidb['boxes'].astype(np.float32)

    return labels, rois

def _get_image_blob_augment(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_infos = [[] for _ in xrange(num_images)]
    im_scales = np.zeros([num_images, 2],dtype = np.float32)

    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert (im.shape[0] > 0)
        
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        target_size = cfg.TRAIN.SCALES[scale_inds[0]]

        im_info = np.array([target_size, target_size], dtype = np.float32)
        im_infos[i] = im_info

        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, im_info)

        im_scales[i,:]=im_scale
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, im_scales, im_infos


def _project_im_rois(im_rois, im_scale_factor, scale_ind):
    """Project image RoIs into the rescaled training image."""
    target_size = cfg.TRAIN.SCALES[scale_ind]
    rois = np.zeros([np.shape(im_rois)[0], np.shape(im_rois)[1]],dtype = np.float32)
    rois[:,0::2] = im_rois[:,0::2] * im_scale_factor[0]
    rois[:,1::2] = im_rois[:,1::2] * im_scale_factor[1]

    return rois


def _vis_minibatch(im_blob, rois_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(im_blob.shape[0]):
        
        rois_inds = rois_blob[:, -1]

        inds = np.where(rois_inds == i)[0]
        rois = rois_blob[inds, :] 
        
        im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = rois[-1]
        print rois
        plt.imshow(im)
        
        for j in xrange(rois.shape[0]):
            roi = rois[j]

            plt.gca().add_patch(
                plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
                )
        plt.show()

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
    im_blob, im_scales, keeps, im_infos, crop_infos, expension_infos = _get_image_blob_augment(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 6), dtype=np.float32)
    im_info_blob = np.zeros((0, 2), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)

    all_overlaps = []
    for im_i in xrange(num_images):
        labels, im_rois = _get_rois(roidb[im_i], keeps[im_i],  crop_infos[im_i], expension_infos[im_i])
            
        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i,:], random_scale_inds[0], expension_infos[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
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

def _get_rois(roidb, keep, crop_info, expension_info):
    """Get the gt RoIs."""
    labels = roidb['gt_classes']
    labels = labels[keep]
    rois = roidb['boxes']
    rois = rois[keep,:].astype(np.float32)
    rois[:,0::2] = rois[:,0::2] - crop_info[0]
    rois[:,1::2] = rois[:,1::2] - crop_info[2]

    if cfg.TRAIN.MOREAUGMENT:
        rois[:,0::2] = rois[:,0::2] + expension_info[1]
        rois[:,1::2] = rois[:,1::2] + expension_info[0]

    return labels, rois

def _get_image_blob_augment(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    crop_ratios = cfg.TRAIN.CROPS
    num_images = len(roidb)
    processed_ims = []
    keeps = [[] for _ in xrange(num_images)]
    im_infos = [[] for _ in xrange(num_images)]
    crop_infos = [[] for _ in xrange(num_images)]
    expension_infos = [[] for _ in xrange(num_images)]
    im_scales = np.zeros([num_images, 2],dtype = np.float32)

    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert (im.shape[0] > 0)
        
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        if cfg.TRAIN.MOREAUGMENT:
            flag = True
            while flag:
                agment_ratio = crop_ratios[npr.randint(0, high=len(crop_ratios),size=1)[0]]
                reshape_flag = npr.randint(0, 2, size=1)[0]
                gt_boxes_clip = np.array(roidb[i]['boxes'], dtype = np.float32)    

                im_shape = im.shape[0:2]
                patch_size_x = im_shape[0] * agment_ratio
                patch_size_y = im_shape[1] * agment_ratio
                if reshape_flag > 0:
                    if im_shape[0] > im_shape[1]:
                        patch_size_x = patch_size_y
                    else:
                        patch_size_y = patch_size_x

                tl_x = npr.randint(0, high= int(im_shape[0] - patch_size_x) + 1, size=1)[0]
                tl_y = npr.randint(0, high= int(im_shape[1] - patch_size_y) + 1, size=1)[0]
                
                crop_info = np.array([tl_y, tl_y + patch_size_y, tl_x, tl_x + patch_size_x], dtype = np.int32)

                im_crop = im[crop_info[2]: crop_info[3], crop_info[0]: crop_info[1], :]
        
                gt_boxes_clip = np.array(roidb[i]['boxes'], dtype = np.float32)         

                gt_boxes_clip[:,0::2] = gt_boxes_clip[:,0::2] - crop_info[0]
                gt_boxes_clip[:,1::2] = gt_boxes_clip[:,1::2] - crop_info[2]

                keep = _filter_boxes(im_crop.shape[0:2], gt_boxes_clip)
                if len(keep) > 0:
                    flag = False

        keeps[i] = keep
        crop_infos[i] = crop_info
        target_size = cfg.TRAIN.SCALES[scale_inds[0]]

        expension_flag = npr.randint(0, 2, size=1)[0]
        if cfg.TRAIN.EXPENSION and expension_flag > 0: 
            im_expension_size = npr.randint(low = np.max(im_crop.shape[0:2]), high= np.max(im_crop.shape[0:2]) * cfg.TRAIN.EXPENSION_SCALE, size=1)[0]
            start_row = npr.randint(0, high= im_expension_size - im_crop.shape[0] + 1, size=1)[0]
            start_col = npr.randint(0, high= im_expension_size - im_crop.shape[1] + 1, size=1)[0]
            expension_infos[i] = [start_row, start_col, start_row + im_crop.shape[0], start_col + im_crop.shape[1]]

            if cfg.TRAIN.EXPENSION_RAND: 
                im_crop_shape = im_crop.shape[0:2]
                im_expension = cv2.resize(im_crop, None, None, fx = float(im_expension_size) / im_crop_shape[1], \
                    fy = float(im_expension_size) / im_crop_shape[0], interpolation = cv2.INTER_LINEAR)
                im_expension = npr.permutation(im_expension.flatten()).reshape((im_expension_size, im_expension_size, 3))
            else:
                im_expension = np.zeros((im_expension_size, im_expension_size, 3), dtype = np.float32)

            im_expension[start_row: start_row + im_crop.shape[0], start_col: start_col + im_crop.shape[1], :] = im_crop
            im_crop = im_expension
        else:
            expension_infos[i] = [0, 0, im_crop.shape[0], im_crop.shape[1]]
   
        im_info = np.array([target_size, target_size], dtype = np.float32)
        im_infos[i] = im_info

        if cfg.TRAIN.COLORDISTORATION:
            im_crop = Image.fromarray(im_crop.astype(np.uint8))  
            enhance_ind = npr.randint(0, 3, size=1)[0]
            enhance_scale = npr.random() + cfg.TRAIN.COLOR_ENHANCE_LO
            if enhance_ind == 0:
                brightness = ImageEnhance.Brightness(im_crop) 
                im_crop = brightness.enhance(enhance_scale)
            elif enhance_ind == 1:
                contrast = ImageEnhance.Contrast(im_crop)  
                im_crop = contrast.enhance(enhance_scale) 
            else:
                color = ImageEnhance.Color(im_crop)  
                im_crop = color.enhance(enhance_scale) 
            im_crop = np.array(im_crop, dtype=np.float32)

        im_crop, im_scale = prep_im_for_blob(im_crop, cfg.PIXEL_MEANS, im_info)

        im_scales[i,:]=im_scale
        processed_ims.append(im_crop)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, im_scales, keeps, im_infos, crop_infos, expension_infos

def _filter_boxes(im_shape, gt_boxes):
    gt_boxes_clip = copy.deepcopy(gt_boxes)
    gt_boxes_clip[:, 0] = np.maximum(gt_boxes_clip[:, 0], 0)
    gt_boxes_clip[:, 1] = np.maximum(gt_boxes_clip[:, 1], 0)
    gt_boxes_clip[:, 2] = np.minimum(gt_boxes_clip[:, 2], im_shape[1] - 1)
    gt_boxes_clip[:, 3] = np.minimum(gt_boxes_clip[:, 3], im_shape[0] - 1)
    overlaps = bbox_overlaps_gt(gt_boxes_clip.astype(np.float), gt_boxes.astype(np.float))

    area_ratios = np.diag(overlaps)
    keep = np.where(area_ratios > cfg.TRAIN.GT_OVERLAP)[0]

    # c_hs = (gt_boxes[:,1] + gt_boxes[:,3]) * 0.5
    # c_ws = (gt_boxes[:,0] + gt_boxes[:,2]) * 0.5
    # keep = np.where((c_hs >= 0) & (c_hs < im_shape[0]) & (c_ws >= 0) & (c_ws < im_shape[1]))[0]
    return keep

def _project_im_rois(im_rois, im_scale_factor, scale_ind, expension_info):
    """Project image RoIs into the rescaled training image."""
    target_size = cfg.TRAIN.SCALES[scale_ind]
    rois = np.zeros([np.shape(im_rois)[0], np.shape(im_rois)[1]],dtype = np.float32)
    rois[:,0::2] = im_rois[:,0::2] * im_scale_factor[0]
    rois[:,1::2] = im_rois[:,1::2] * im_scale_factor[1]

    rois[:, 0] = np.maximum(rois[:, 0], expension_info[1] * im_scale_factor[0])
    rois[:, 1] = np.maximum(rois[:, 1], expension_info[0] * im_scale_factor[1])
    rois[:, 2] = np.minimum(rois[:, 2],  expension_info[3] * im_scale_factor[0] -1)
    rois[:, 3] = np.minimum(rois[:, 3],  expension_info[2] * im_scale_factor[1]  -1)

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

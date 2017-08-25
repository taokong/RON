# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Tao Kong, 2016-11-22
# --------------------------------------------------------
from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import clip_boxes, filter_boxes
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt


def _get_image_blob(ims, target_size):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_infos(ndarray): a data blob holding input size pyramid
    """
    processed_ims = []
    for im in ims:
        im = im.astype(np.float32, copy = False)
        im = im - cfg.PIXEL_MEANS
        im_shape = im.shape[0:2]
        im = cv2.resize(im, None, None, fx = float(target_size) / im_shape[1], \
            fy = float(target_size) / im_shape[0], interpolation = cv2.INTER_LINEAR)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
   
    return blob

def _get_blobs(ims, target_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data']= _get_image_blob(ims, target_size)

    return blobs

def im_detect(net, ims):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs = _get_blobs(ims, target_size = cfg.TEST.SCALES[0])

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

    pred_boxes7 = blobs_out['rois7']
    scores7 = blobs_out['scores7']

    pred_boxes6 = blobs_out['rois6']
    scores6 = blobs_out['scores6']

    pred_boxes5 = blobs_out['rois5']
    scores5 = blobs_out['scores5']

    pred_boxes4 = blobs_out['rois4']
    scores4 = blobs_out['scores4']

    pred_boxes = np.zeros((cfg.TEST.BATCH_SIZE, 0, 4), dtype=np.float32)
    scores = np.zeros((cfg.TEST.BATCH_SIZE, 0, scores7.shape[-1]), dtype=np.float32) 

    scores = np.concatenate((scores, scores7), axis = 1)
    scores = np.concatenate((scores, scores6), axis = 1)
    scores = np.concatenate((scores, scores5), axis = 1)
    scores = np.concatenate((scores, scores4), axis = 1)

    pred_boxes = np.concatenate((pred_boxes, pred_boxes7), axis = 1)
    pred_boxes = np.concatenate((pred_boxes, pred_boxes6), axis = 1)
    pred_boxes = np.concatenate((pred_boxes, pred_boxes5), axis = 1)
    pred_boxes = np.concatenate((pred_boxes, pred_boxes4), axis = 1)     

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.5):
    """Visual debugging of detections."""

    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(5, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]

        if score > thresh:
            
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()



def test_net(net, imdb, vis = 0):
    """Test RON network on an image database."""
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    for i in xrange(0, num_images, cfg.TEST.BATCH_SIZE): 
        _t['misc'].tic()
        ims = []
        for im_i in xrange(cfg.TEST.BATCH_SIZE):
            im = cv2.imread(imdb.image_path_at(i+im_i))
            ims.append(im)
        _t['im_detect'].tic()
        batch_scores, batch_boxes = im_detect(net,ims)
        _t['im_detect'].toc()

        for im_i in xrange(cfg.TEST.BATCH_SIZE):
            im = ims[im_i]
            scores = batch_scores[im_i]
            boxes = batch_boxes[im_i]

            # filter boxes according to prob scores
            keeps = np.where(scores[:,0] > cfg.TEST.PROB)[0]
            scores = scores[keeps, :]
            boxes = boxes[keeps, :]

            # change boxes according to input size and the original image size
            im_shape = im.shape[0:2]
            im_scales = float(cfg.TEST.SCALES[0]) / np.array(im_shape)

            boxes[:, 0::2] =  boxes[:, 0::2] / im_scales[1]
            boxes[:, 1::2] =  boxes[:, 1::2] / im_scales[0]

            # filter boxes with small sizes
            boxes = clip_boxes(boxes, im_shape)
            keep = filter_boxes(boxes, cfg.TEST.RON_MIN_SIZE )
            scores = scores[keep,:]
            boxes = boxes[keep, :]

            scores = np.tile(scores[:, 0], (imdb.num_classes, 1)).transpose() * scores

            for j in xrange(1, imdb.num_classes):
                inds = np.where(scores[:, j] > cfg.TEST.DET_MIN_PROB)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, :]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]
                if len(keep) > cfg.TEST.BOXES_PER_CLASS:
                    cls_dets = cls_dets[:cfg.TEST.BOXES_PER_CLASS,:]
                all_boxes[j][i+im_i] = cls_dets
            
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)
            _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)



   

"""
RON: Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from fast_rcnn.bbox_transform import clip_boxes, filter_boxes

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def detect(net, im):
    # Detect all object classes and regress object bounds
    ims = []
    ims.append(im)
    scores, boxes = im_detect(net, ims)
    scores = scores[0]
    boxes = boxes[0]
    # filter boxes according to prob scores
    keeps = np.where(scores[:,0] > cfg.TEST.PROB)[0]
    scores = scores[keeps, :]
    boxes = boxes[keeps, :]

    # change boxes according to input size and the original image size
    im_shape = np.array(im.shape[0:2])
    im_scales = float(cfg.TEST.SCALES[0]) / im_shape
    
    boxes[:, 0::2] =  boxes[:, 0::2] / im_scales[1]
    boxes[:, 1::2] =  boxes[:, 1::2] / im_scales[0]

    # filter boxes with small sizes
    boxes = clip_boxes(boxes, im_shape)
    keeps = filter_boxes(boxes, cfg.TEST.RON_MIN_SIZE )
    scores = scores[keeps,:]
    boxes = boxes[keeps, :]

    scores = np.tile(scores[:, 0], (len(CLASSES), 1)).transpose() * scores

    return scores, boxes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a grasp network')
    parser.add_argument('--video', dest='video', help='video used to test',
                        default='', type=str)
    parser.add_argument('--model', dest='model', help='model used to test',
                        default='', type=str)
    parser.add_argument('--weights', dest='weights', help='weights used to test',
                        default='', type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    _t = {'im_detect' : Timer()}

    prototxt = os.path.join(cfg.ROOT_DIR, args.model)
    caffemodel = os.path.join(cfg.ROOT_DIR, args.weights)
    cfg.MINANCHOR = 24
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    top_N = 30
    max_score = 0.6

    videoCam = cv2.VideoCapture(os.path.join(cfg.ROOT_DIR, args.video))
    # videoCam = cv2.VideoCapture(0)

    while cv2.waitKey(1) != 0x20:
        success, frame = videoCam.read()
        
        if not success:
            print "Error getting frame"
            break
        _t['im_detect'].tic()
        scores, boxes = detect(net, frame)
        _t['im_detect'].toc()
        for j in xrange(1, len(CLASSES)):
            inds = np.where(scores[:, j]> max_score)[0]
            cls_boxes = boxes[inds, :]
            cls_scores = scores[inds, j]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)

            cls_dets = cls_dets[keep, :]

            if len(keep) > top_N:
                keep = keep[:top_N]

            for k in xrange(len(keep)):
                bbox = cls_dets[k, 0:4]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 3)
                cv2.putText(frame, CLASSES[j], (bbox[0], bbox[1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 2)
        cv2.imshow("RON detection results", frame)


    videoCam.release()
    cv2.destroyAllWindows()


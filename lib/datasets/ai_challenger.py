# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import json
import cv2
from matplotlib import pyplot

class ai_challenger(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'ai_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path

        self._classes = ('__background__', # always index 0
                         'human')

        mid_str = 'ai_challenger_keypoint_'+image_set+'_'+year

        ann_string = 'keypoint_'+ image_set + '_annotations_'+ year + '.json'
        image_forder_string = 'keypoint_'+image_set+'_images_'+ year

        self.annotation_file = os.path.join(self._data_path, mid_str, ann_string)

        self.image_forder = os.path.join(self._data_path, mid_str, image_forder_string)

        self._dataset = json.load(open(self.annotation_file, 'r'))

        self._box_heights = []
        self._box_widths = []
        self._box_height2widths = []

        self._image_ext = '.jpg'
        self._image_index = [ann['image_id'] for ann in self._dataset]
        # Default to roidb handler
        self._roidb = self.gt_roidb()

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.image_forder, index+'.jpg')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return  image_path


    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return '/media/kongtao/448A94818A947162/dataset/ai_challenge'

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file_imageindex = os.path.join(self.cache_path, self.name + '_image_index.pkl')
        if os.path.exists(cache_file_imageindex):
            with open(cache_file_imageindex, 'rb') as fid:
                self._image_index = cPickle.load(fid)

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb


        im_shapes = self._get_shapes()
        print np.shape(im_shapes)

        gt_roidb0 = [self._load_annotation(index, im_shapes[index][1], im_shapes[index][0])
                    for index in xrange(len(self._image_index))]

        has_obj_num = 0
        for index in xrange(len(self._image_index)):
            if gt_roidb0[index]['boxes'].shape[0] > 0:
                has_obj_num += 1

        gt_roidb = [1 for _ in xrange(has_obj_num)]
        obj_index = 0
        image_index = [[] for _ in xrange(has_obj_num)]
        for index in xrange(len(self._image_index)):
            if gt_roidb0[index]['boxes'].shape[0] > 0:
                gt_roidb[obj_index] = gt_roidb0[index]
                image_index[obj_index] = self._image_index[index]
                obj_index = obj_index + 1

        self._image_index = image_index

        for i in xrange(len(self._image_index)):
            gt_roidb[i]['image'] = self.image_path_at(i)


        with open(cache_file_imageindex, 'wb') as fid:
            cPickle.dump(self._image_index, fid, cPickle.HIGHEST_PROTOCOL)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)

        print 'wrote gt roidb to {}'.format(cache_file)


        return gt_roidb


    def _load_annotation(self, index, height, width):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        if(index % 1000 == 0):
            print 'loading annotations:', index

        bboxes_json = self._dataset[index]
        bboxes = bboxes_json['human_annotations']
        keypoints = bboxes_json['keypoint_annotations']

        boxes = np.zeros((0, 4), dtype=np.uint16)
        key_pointss = np.zeros((0, 3*14), dtype=np.uint16)
        gt_classes = []
        overlaps = np.zeros((0, 2), dtype=np.float32)
        # "Seg" area for pascal is just the box area

        assert len(bboxes) > 0, "has no object "+ index
        # Load object bounding boxes into a data frame.
        for ix in xrange(len(bboxes)):
            human_id = 'human'+str(ix+1)
            human_bbox = bboxes[human_id]
            human_keypoint = keypoints[human_id]
            cls = 1

            if(human_bbox[2] > human_bbox[0] and \
                human_bbox[3] > human_bbox[1] and \
               human_bbox[0] >=0 and \
               human_bbox[1] >=0 and \
               human_bbox[2] < width and \
               human_bbox[3] < height):

                self._box_heights.append(float(human_bbox[3] - human_bbox[1] + 1.0)/height)
                self._box_widths.append(float(human_bbox[2] - human_bbox[0] + 1.0)/width)
                self._box_height2widths.append(float(human_bbox[3] - human_bbox[1] + 1.0) / \
                                               float(human_bbox[2] - human_bbox[0] + 1.0))

                boxes = np.vstack((boxes, human_bbox))
                key_pointss = np.vstack((key_pointss, human_keypoint))
                gt_classes.append(cls)
                overlaps = np.vstack((overlaps, [0, 1.0]))

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'key_points': key_pointss,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}


    def visualization(self, i):

        image_path = self.image_path_at(i)
        image = cv2.imread(image_path)

        bboxes = self.roidb[i]['boxes']
        key_points_all = self.roidb[i]['key_points']

        for i in xrange(np.shape(bboxes)[0]):
            bbox = bboxes[i, :]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)

            key_points = key_points_all[i, :]

            xs = key_points[0::3]
            ys = key_points[1::3]
            vs = key_points[2::3]

            color = (0, 255, 0)
            for j, v in enumerate(vs):
                if(v < 3):
                    cv2.circle(image, (xs[j],ys[j]), 5, color)
            if(vs[0] < 3 and vs[1] < 3):
                cv2.line(image, (xs[0], ys[0]),(xs[1], ys[1]), color)
            if(vs[1] < 3 and vs[2] < 3):
                cv2.line(image, (xs[1], ys[1]),(xs[2], ys[2]), color)

            if(vs[3] < 3 and vs[4] < 3):
                cv2.line(image, (xs[3], ys[3]),(xs[4], ys[4]), color)
            if(vs[4] < 3 and vs[5] < 3):
                cv2.line(image, (xs[4], ys[4]),(xs[5], ys[5]), color)

            if(vs[0] < 3 and vs[3] < 3):
                cv2.line(image, (xs[0], ys[0]),(xs[3], ys[3]), color)

            if(vs[6] < 3 and vs[7] < 3):
                cv2.line(image, (xs[6], ys[6]),(xs[7], ys[7]), color)
            if(vs[7] < 3 and vs[8] < 3):
                cv2.line(image, (xs[7], ys[7]),(xs[8], ys[8]), color)

            if(vs[9] < 3 and vs[10] < 3):
                cv2.line(image, (xs[9], ys[9]),(xs[10], ys[10]), color)
            if(vs[10] < 3 and vs[11] < 3):
                cv2.line(image, (xs[10], ys[10]),(xs[11], ys[11]), color)

            if(vs[12] < 3 and vs[13] < 3):
                cv2.line(image, (xs[12], ys[12]),(xs[13], ys[13]), color)

            if(vs[3] < 3 and vs[9] < 3):
                cv2.line(image, (xs[3], ys[3]),(xs[9], ys[9]), color)
            if(vs[0] < 3 and vs[6] < 3):
                cv2.line(image, (xs[0], ys[0]),(xs[6], ys[6]), color)

        cv2.imshow("visualization", image)
        cv2.waitKey(0)



    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.ai_challenger import ai_challenger
    d = ai_challenger('validation', '20170911')
    res = d.roidb

    print d._box_height2widths
    pyplot.hist(d._box_height2widths,20)
    pyplot.show()

    cache_heights_file = os.path.join(d.cache_path, d.name + '_bbox_heighs.pkl')
    with open(cache_heights_file, 'wb') as fid:
            cPickle.dump(d._box_heights, fid, cPickle.HIGHEST_PROTOCOL)

    cache_widths_file = os.path.join(d.cache_path, d.name + '_bbox_widths.pkl')
    with open(cache_widths_file, 'wb') as fid:
            cPickle.dump(d._box_widths, fid, cPickle.HIGHEST_PROTOCOL)

    cache_height2widths_file = os.path.join(d.cache_path, d.name + '_bbox_height2widths.pkl')
    with open(cache_height2widths_file, 'wb') as fid:
            cPickle.dump(d._box_height2widths, fid, cPickle.HIGHEST_PROTOCOL)

    for i in xrange(20):
        d.visualization(i*10)

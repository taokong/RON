# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and modified by Tao Kong
# --------------------------------------------------------

"""Train RON network."""

import caffe
from fast_rcnn.config import cfg
from utils.timer import Timer
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """
    A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)

        if model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(model)
            self.solver.net.copy_from(model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """
        Take a snapshot of the network.
        """
        net = self.solver.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


        infix = '_' + str(cfg.TRAIN.SCALES[0])
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'
    print 'done'
    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir,
              model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir, model=model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'

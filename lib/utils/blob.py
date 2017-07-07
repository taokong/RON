# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Tao Kong
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
from fast_rcnn.config import cfg
def im_list_to_blob(ims):
    """
    Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, im_info):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape[0:2]
    
    fy_scale, fx_scale =  im_info / im_shape
    im = cv2.resize(im, None, None, fx=fx_scale, fy=fy_scale,
                    interpolation=cv2.INTER_LINEAR)  
 
    im_scales = np.array([fx_scale, fy_scale])    
    return im, im_scales

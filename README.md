# RON: Reverse Connection with Objectness Prior Networks for Object Detection
RON is a state-of-the-art visual object detection system for efficient object detection framework. The code is modified from py-faster-rcnn and caffe. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our arXiv paper and our slide.

### Citing RON

If you find RON useful in your research, please consider citing:

    @inproceedings{KongtCVPR2017,
        Author = {Tao Kong, Fuchun Sun, Anbang Yao, Huaping Liu, Yurong Chen and Ming Lu},
        Title = {RON: Reverse Connection with Objectness Prior Networks for Object Detection},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        Year = {2017}
    }
    

### PASCAL VOC detection results

Method         | VOC 2007 mAP | VOC 2012 mAP 
-------------- |:------------:|:------------:
Fast R-CNN     |   70.0%      |   68.4%        
Faster R-CNN   |   73.2%      |   70.4%        
SSD300         |   72.1%      |   70.3%
SSD500         |   75.1%      |   73.1%
RON320         |   74.2%      |   71.7%        
RON384         |   75.4%      |   73.0%        


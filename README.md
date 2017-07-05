# RON: Reverse Connection with Objectness Prior Networks for Object Detection

RON is a state-of-the-art visual object detection system for efficient object detection framework. The code is modified from py-faster-rcnn. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our arXiv paper.

### Citing RON

If you find RON useful in your research, please consider citing:

    @inproceedings{KongtCVPR2017,
        Author = {Tao Kong, Fuchun Sun, Anbang Yao, Huaping Liu, Yurong Chen and Ming Lu},
        Title = {RON: Reverse Connection with Objectness Prior Networks for Object Detection},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        Year = {2017}
    }
    

### PASCAL VOC detection results

Method         | VOC 2007 mAP | VOC 2012 mAP | Input resolution
-------------- |:------------:|:------------:|:----------------
Fast R-CNN     |   70.0%      |   68.4%      |  1000*600     
Faster R-CNN   |   73.2%      |   70.4%      |  1000*600
SSD300         |   72.1%      |   70.3%      |  300*300
SSD500         |   75.1%      |   73.1%      |  500*500
RON320         |   74.2%      |   71.7%      |  320*320
RON384         |   75.4%      |   73.0%      |  384*384

### MS COCO detection results

Method         | Training data | AP(0.50-0.95)| Input resolution
-------------- |:-------------:|:------------:|:----------------
Faster R-CNN   |   trainval    |   21.9%      |  1000*600
SSD500         |   trainval35k |   24.4%      |  500*500
RON320         |   trainval    |   23.6%      |  320*320
RON384         |   trainval    |   25.4%      |  384*384

Note: SSD300 and SSD500 are the original SSD model from [SSD](https://arxiv.org/pdf/1512.02325v2.pdf).

## TODO
Codes are coming soon.


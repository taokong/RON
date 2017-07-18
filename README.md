# RON: Reverse Connection with Objectness Prior Networks for Object Detection

RON is a state-of-the-art visual object detection system for efficient object detection framework. 
The code is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 
You can use the code to train/evaluate a network for object detection task. 
For more details, please refer to our CVPR paper.

### Citing RON

If you find RON useful in your research, please consider citing:

    @inproceedings{KongtCVPR2017,
        Author = {Tao Kong, Fuchun Sun, Anbang Yao, Huaping Liu, Ming Lu, Yurong Chen},
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


### RON Installation 

0. Clone the RON repository
    ```
    git clone https://github.com/taokong/RON.git

    ```
1. Build Caffe and pycaffe

    ```
    cd $RON_ROOT/
    git clone https://github.com/taokong/caffe-ron.git
    cd caffe-ron
    make -j8 && make pycaffe
    *this version use CUDNN for efficiency, so make sure that "USE_CUDNN := 1" in the Makefile.config file.
    ```

2. Build the Cython modules
    ```
    cd $RON_ROOT/lib
    make
    ```
    
3. installation for training and testing models on PASCAL VOC dataset

    3.0 The PASCAL VOC dataset has the basic structure:
    
        $VOCdevkit/                           # development kit
        $VOCdevkit/VOCcode/                   # VOC utility code
        $VOCdevkit/VOC2007                    # image sets, annotations, etc.
        
    3.1 Create symlinks for the PASCAL VOC dataset
    
        cd $RON_ROOT/data
        ln -s $VOCdevkit VOCdevkit2007
        ln -s $VOCdevkit VOCdevkit2012

4. Test with PASCAL VOC datset

    Now we provide two models for testing the pascal voc 2007 test dataset. To use demo you need to download the pretrained RON model, please download the model manually from [BaiduYun](https://pan.baidu.com/s/1o8QEwu2), and put it under `$data/RON_models`.
    
    4.0 The original model as introduced in the RON paper: 
    
        ./test_voc07.sh
        # The final result of the model should be 74.2% mAP.
        
    4.1 A lite model we make some optimization after the original one:

        ./test_voc07_reduced.sh
        # The final result of the model should be 74.1% mAP.

5. Train with PASCAL VOC datset

    Please download ImageNet-pre-trained VGG models manually from [BaiduYun](https://pan.baidu.com/s/1c2xm2U8), and put them into `$data/ImageNet_models`. Then everything is done, you could train your own model.

    5.0 The original model as introduced in the RON paper: 
    
        ./train_voc.sh
        
    5.1 A lite model we make some optimization after the original one:

        ./train_voc_reduced.sh

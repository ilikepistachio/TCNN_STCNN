Project website: [http://crcv.ucf.edu/projects/TCNN/](http://crcv.ucf.edu/projects/TCNN/)
# Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos

By Rui Hou, Chen Chen and Mubarak Shah

### Abstract:

We propose an end-to-end deep network called Tube Convolutional Neural Network (T-CNN) for action detection in videos. The proposed architecture is a unified deep network that is able to recognize and localize action based on 3D convolution features. A video is first divided into equal length clips and next for each clip a set of tube proposals are generated based on 3D Convolutional Network (ConvNet) features. Finally, the tube proposals of different clips are linked together employing network flow and spatio-temporal action detection is performed using these linked video proposals. Extensive experiments on several video datasets demonstrate the superior performance of TCNN for classifying and localizing actions

# (News) An End-to-end 3D Convolutional Neural Network for Action Detection and Segmentation in Videos

### Abstract:
Deep learning has been demonstrated to achieve excellent results for image classification and object detection. However,
the impact of deep learning on video analysis (e.g. action detection and recognition) has not been that significant due to complexity of
video data and lack of annotations. In addition, training deep neural networks on large scale video datasets is extremely
computationally expensive. Previous convolutional neural networks (CNNs) based video action detection approaches usually consist of
two major steps: frame-level action proposal generation and association of proposals across frames. Also, most of these methods
employ two-stream CNN framework to handle spatial and temporal features separately. In this paper, we propose an end-to-end 3D
CNN for action detection and segmentation in videos. The proposed architecture is a unified deep network that is able to recognize and
localize action based on 3D convolution features. A video is first divided into equal length clips and next for each clip a set of tube
proposals are generated based on 3D CNN features. Finally, the tube proposals of different clips are linked together and
spatio-temporal action detection is performed using these linked video proposals. This top-down action detection approach explicitly
relies on a set of good tube proposals to perform well and training the bounding box regression usually requires a large number of
annotated samples. To remedy this, we further extend the 3D CNN to an encoder-decoder structure and formulate the localization
problem as action segmentation. The foreground regions (i.e. action regions) for each frame are segmented first then the segmented
foreground maps are used to generate the bounding boxes. This bottom-up approach effectively avoids tube proposal generation by
leveraging the pixel-wise annotations of segmentation. The segmentation framework also can be readily applied to a general problem
of video object segmentation. Extensive experiments on several video datasets demonstrate the superior performance of our approach
for action detection and video object segmentation compared to the state-of-the-arts.

This code has been tested on Ubuntu 16.04 with a single NVIDIA GeForce GTX TITAN Xp graphic card.

[comment]: # ()
Current code is our rough version and we are improving its implementation details, while the current version suffices to run demo, repeat our experimental results, and train your own models.

### License

T-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing:

If you find T-CNN useful, please consider citing:

    @article{hou2017tube,
        title={Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos},
        author={Hou, Rui and Chen, Chen and Shah, Mubarak},
        journal={arXiv preprint arXiv:1703.10664},
        year={2017} }
    
### Installation:
- Hint: please refer to [C3D-v1.0](https://github.com/facebook/C3D/tree/master/C3D-v1.0) and [Caffe](https://github.com/BVLC/caffe) for more details about compilation such as making your own Makefile.config
- For the sake of using our code smoothly, please first get familiar with [C3D](https://github.com/facebook/C3D).

### Run demo:
- This demo is designed to let users to have a quick try of tcnn feature extraction.
- More details of this demo:
1. we provide input data in `data/jhmdb`. You can prepare the download the dataset by `bash prepre_jhmdb.sh`
2. output results will be stored in `data/jhmdb/results`

### Reproduce results on J-HMDB dataset:
- Data preparation
1. Move to the root directory of this project. e.g. `cd ~/tcnn`
2. Run the script to download and extract frames, annotations and splits of dataset `bash prepre_jhmdb.sh`
3. The generated dataset is located at `./data/jhmdb`

- T-CNN network prediction
1. Download the trained model
2. the last layer of our trained model has 22 nodes corresponding to 21 possible frame-level classes(from the first to the last: background, action1-22)
3. Run `python tpn_eval.py` for bounding boxes on each frame.

### Train your own model:
- Prepare pre-trained model as init: as explained in the paper, we use weights in sports1m model (`wget www.cs.ucf.edu/~rhou/files/c3d_pretrain_model`) to init our network.
- Run `bash tpn_train.sh` for training TPN.
- Run `bash cls_train.sh` for recognition network.

### ST-CNN demo:
- Run `python demo.py`

### ST-CNN training demo:
- Run `bash demo_train.sh`

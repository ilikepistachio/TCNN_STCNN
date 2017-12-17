Project website: [http://crcv.ucf.edu/projects/TCNN/](http://crcv.ucf.edu/projects/TCNN/)

# Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos

By Rui Hou, Chen Chen and Mubarak Shah

### Introduction:

We propose an end-to-end deep network called Tube Convolutional Neural Network (T-CNN) for action detection in videos. The proposed architecture is a unified deep network that is able to recognize and localize action based on 3D convolution features. A video is first divided into equal length clips and next for each clip a set of tube proposals are generated based on 3D Convolutional Network (ConvNet) features. Finally, the tube proposals of different clips are linked together employing network flow and spatio-temporal action detection is performed using these linked video proposals. Extensive experiments on several video datasets demonstrate the superior performance of TCNN for classifying and localizing actions in b

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
- This demo is designed to let users to have a quick try of CDC feature extraction.
- More details of this demo:
1. we provide input data in `demo/data/window` along with input data list file `demo/data/test.lst`
2. each input data sample is a 8-frames long window. In order to directly reuse `VIDEO_DATA` data format developed in [C3D-v1.0](https://github.com/facebook/C3D/tree/master/C3D-v1.0), each of our input data is stored in bin format and consists of pixel values stacked over time. We provide an example code for generating such bin file on UCF Sports data set in the next section.
3. run the demo: `cd demo; ./xfeat.sh;`
4. output results will be stored in `demo/feat`

### Reproduce results on J-HMDB dataset:
- Data preparation
1. Move to the root directory of this project. e.g. `cd ~/tcnn`
2. Run the script to download and extract frames, annotations and splits of dataset `bash prepre_jhmdb.sh`
3. The generated dataset is located at `./data/jhmdb`

- T-CNN network prediction
1. `cd THUMOS14/test` and you will see needed files for using CDC network to do prediction (i.e. feature extraction of the last layer) and outputs will be stored in `feat`
2. the trained model used for feature extraction is `$TCNN_ROOT/models/ucf_sports/c3d_seg_iter_33200.caffemodel`
3. the last layer of our trained model has 10 nodes corresponding to 11 possible frame-level classes(from the first to the last: background, action1-10)


### Train your own model:
- Prepare pre-trained model as init: as explained in the paper, we use weights in sports1m model (`model/sports1m_C3D/conv3d_deepnetA_sport1m_iter_1900000`) to init our CDC network.
- Run 'toi_rec_train.sh' for training.

Please find our caffe implementation at [ruihou/mtcnn](https://github.com/ruihou/mtcnn)


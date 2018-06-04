#!/usr/bin/env bash
wget http://www.cs.ucf.edu/~rhou/files/c3d_pretrain_model
export PYTHONPATH=$PWD
~/caffe/.build_release/tools/caffe.bin train -gpu 0 \
    -solver ./models/davis/c3d_deconv_solver.prototxt \
    -weights c3d_pretrain_model

echo "Show trained results"
python demo.py davis_8_iter_2000.caffemodel
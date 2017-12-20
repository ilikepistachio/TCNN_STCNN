#!/usr/bin/env bash
export PYTHONPATH=/home/rhou/videoflow
~/caffe/build/tools/caffe train -gpu 0 -solver ./models/c3d_solver.prototxt -weights /home/rhou/models/ucf101_c3d_iter_2715.caffemodel

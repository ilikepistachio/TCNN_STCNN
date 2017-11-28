#!/usr/bin/env bash
export PYTHONPATH=/home/rhou/videoflow
~/caffe/build/tools/caffe train -gpu 0 -solver ./models/ucf_101/toi_frame_solver.prototxt -weights c3d_pretrain_model

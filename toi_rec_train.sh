#!/usr/bin/env bash
export PYTHONPATH=/home/rhou/videoflow
~/caffe/build/tools/caffe train -gpu 0 -solver ./models/ucf_sports/toi_rec_solver.prototxt -weights /home/rhou/models/sports_rec_300_400_iter_24000.caffemodel #/home/rhou/models/sports_frame_300_400_pre_iter_4000.caffemodel

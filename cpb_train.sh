#!/usr/bin/env bash
export PYTHONPATH=/home/rhou/videoflow
~/caffe/build/tools/caffe train -gpu 0 -solver ./models/jhmdb/toi_rec_solver.prototxt -weights /home/rhou/models/jhmdb_rec_300_400_iter_10000.caffemodel

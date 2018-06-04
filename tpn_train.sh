#!/usr/bin/env bash
export PYTHONPATH=/home/rhou/tcnn
~/caffe/build/tools/caffe train -gpu 0 -solver ./models/jhmdb/tpn_rec_solver.prototxt -weights /home/rhou/c3d_pretrain_model
'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.jhmdb import jhmdb
import numpy as np
import os

class DataEval():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = jhmdb('eval', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    [clips, gt_bboxes, gt_label, vid_name, is_last] \
      = self.dataset.next_val_video()

    tois = np.load('data/jhmdb/tpn/{}/bboxes.npy'.format(vid_name)) * 1.25 / 16
    num_frames = clips.shape[0]
    scores = np.zeros((num_frames))
    for i in xrange(num_frames - self._depth + 1):
      batch_clip = np.expand_dims(clips[i : i + self._depth].transpose(3, 0, 1, 2), axis=0)
      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      curr_bbox = np.mean(tois[i * self._depth: (i + 1) * self._depth],
                          axis=0)
      batch_tois = np.expand_dims(curr_bbox, axis=0)
      self._net.blobs['tois'].data[...] = batch_tois.astype(np.float32,
                                                            copy=False)
      self._net.forward()
      s = self._net.blobs['loss'].data[...]
      scores[i: i + self._depth] += s

    pred = scores.argmax(axis=1)

    return is_last

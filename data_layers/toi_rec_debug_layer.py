'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.thumos14 import thumos14
import numpy as np


class RecDataLayer():
  def __init__(self):
    net = '/home/rhou/videoflow/models/toi_rec_eval.prototxt'
    model = '/home/rhou/videoflow/c3d_pretrain_model'
    self._batch_size = 1
    self._depth = 16
    self._height = 300
    self._width = 400
    self._d = thumos14('train', [self._height, self._width],
                             '/home/rhou/thumos14')
    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TRAIN)

  def forward(self):
    [batch_clip, batch_labels, batch_bboxes, _] \
      = self._d.next_specific_batch('v_GolfSwing_g23_c07', 83, 16)
    batch_clip = batch_clip.transpose((0, 4, 1, 2, 3))
    self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                          copy=False)
    self._net.blobs['label'].data[...] = batch_labels.astype(np.float32,
                                                                  copy=False)
    self._net.blobs['tois'].data[...] = batch_bboxes.astype(np.float32,
                                                                 copy=False)
    print(batch_bboxes)
    r = self._net.forward()
    print(r)
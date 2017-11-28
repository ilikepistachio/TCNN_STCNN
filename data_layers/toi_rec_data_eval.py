'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.ucf_sports import UcfSports
import numpy as np
from utils.cython_bbox import bbox_overlaps

class RecDataLayer():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = UcfSports('test', [self._height, self._width],
                             '/home/rhou/ucf_sports')

    self.anchors = self.dataset.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    self._net.blobs['data'].reshape(self._batch_size, 3,
                                    self._depth, self._height, self._width)
    self._net.blobs['tois'].reshape(self._batch_size * 3714, 5)

    [clip, labels, gt_bboxes, is_last] = self.dataset.next_val_video(random=False)

    n = int(np.floor(clip.shape[0] / 8.0))

    result = np.empty((n, 3714, 22))
    for i in xrange(n):
      batch_clip = clip[i * 8 : i * 8 + 8].transpose([3, 0, 1, 2])
      batch_clip = np.expand_dims(batch_clip, axis=0)

      batch_tois = np.hstack((np.zeros((3714, 1)), self.anchors))

      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      self._net.blobs['tois'].data[...] = batch_tois.astype(np.float32,
                                                              copy=False)
      self._net.forward()
      r1 = self._net.blobs['loss'].data[...]
      r2 = self._net.blobs['fc8-1'].data[...]

      result[i] = np.hstack((r1, r2))

    return result, is_last

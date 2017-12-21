'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.jhmdb import jhmdb
import numpy as np
from utils.cython_bbox import bbox_overlaps
from utils.bbox_transform import bbox_transform

class DataEval():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = jhmdb('val', [self._height, self._width], split=1)

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    [video, _, gt_label, vid_name, pred, is_last] = self.dataset.next_rec_video()
    num_frames = video.shape[0]
    n_clips = num_frames / self._depth
    batch_clip = np.empty((n_clips, 3, self._depth, self._height, self._width))
    batch_tois = np.empty((n_clips, 5))
    batch_label = np.empty((1, 1))
    batch_label[0, 0] = gt_label

    for i in xrange(n_clips):
      batch_clip[i] = video[i * self._depth : (i + 1) * self._depth].transpose((3, 0, 1, 2))
      curr_bbox = np.mean(pred[i * self._depth : (i + 1) * self._depth],
                          axis=0) / 16
      batch_tois[i] = np.concatenate((np.ones(1) * i, curr_bbox))

      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      self._net.blobs['tois'].data[...] = batch_tois.astype(np.float32,
                                                            copy=False)
      self._net.forward()
      s = self._net.blobs['loss'].data[...]

    pred = s.argmax()
    return is_last
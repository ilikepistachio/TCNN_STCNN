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

class DataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = jhmdb('train', [self._height, self._width], split=1)
    self.anchors, _, _ = self.dataset.get_anchors()
    self.num_boxes = 16

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # tois.
    top[1].reshape(self._batch_size * self.num_boxes, 5)
    # gt label
    top[2].reshape(1, 1)


  def forward(self, bottom, top):
    [video, gt_bboxes, gt_label, _, _, _] = self.dataset.next_rec_video()
    num_frames = video.shape[0]
    n_clips = num_frames / self._depth
    batch_clip = np.empty((n_clips, 3, self._depth, self._height, self._width))
    batch_tois = np.empty((n_clips, 5))
    batch_label = np.empty((1, 1))
    batch_label[0, 0] = gt_label

    for i in xrange(n_clips):
      batch_clip[i] = video[i * self._depth : (i + 1) * self._depth].transpose((3, 0, 1, 2))
      curr_bbox = np.mean(gt_bboxes[i * self._depth : (i + 1) * self._depth],
                          axis=0) / 16
      batch_tois = np.concatenate((np.ones(1) * i, curr_bbox))

    top[0].reshape(*batch_clip.shape)
    top[1].reshape(*batch_tois.shape)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_tois.astype(np.float32, copy=False)
    top[2].data[...] = batch_label.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
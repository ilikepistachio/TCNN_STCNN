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

class DataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    self.dataset = jhmdb('train', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()
    self.num_boxes = 32

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # Ground truth labels.
    top[1].reshape(self._batch_size * self._depth, 1, self._anchor_dims[0] * self._anchor_dims[1] * self._anchor_dims[2])

  def forward(self, bottom, top):
    [clips, labels, tmp_bboxes, box_idx] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_labels = np.empty((self._batch_size * self._depth, 1, self._anchor_dims[0] * self._anchor_dims[1] * self._anchor_dims[2]))

    for i in xrange(self._depth):
      box = tmp_bboxes[0, :, :]
      gt_bboxes = np.expand_dims((box[i] / 16), axis=0)

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      curr_labels = np.ones(self._anchor_dims[0] *
                            self._anchor_dims[1] *
                            self._anchor_dims[2]) * (-1)
      curr_labels[self.valid_idx[max_overlaps < 0.5]] = 0
      curr_labels[self.valid_idx[max_overlaps > 0.6]] = 1
      curr_labels[self.valid_idx[gt_argmax_overlaps]] = 1
      batch_labels[i, 0] = curr_labels.reshape((self._anchor_dims[1], self._anchor_dims[2], self._anchor_dims[0])).transpose((2, 0, 1)).reshape(-1)


    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_labels.astype(np.float32, copy=False)
    #top[2].data[...] = batch_tois.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
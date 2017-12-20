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
    self.num_boxes = 64

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # tois.
    top[1].reshape(self._batch_size * self.num_boxes, 5)
    # gt label
    top[2].reshape(self._batch_size * self.num_boxes, 1)


  def forward(self, bottom, top):
    [clips, labels, tmp_bboxes, box_idx] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_tois = np.empty((0, 5))
    batch_label = np.empty((0, 1))

    u_i = np.unique(box_idx)
    for i in u_i:
      curr_idx = np.where(box_idx == i)[0]
      box = tmp_bboxes[curr_idx, :, :]
      gt_bboxes = np.mean(box, axis=1) / 16

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))

      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      curr_labels = np.ones(self.anchors.shape[0]) * (-1)
      curr_labels[max_overlaps < 0.4] = 0
      curr_labels[max_overlaps >= 0.6] = labels[0]
      curr_labels[gt_argmax_overlaps] = labels[0]

      fg_inds = np.where(curr_labels > 0)[0]
      num_fg = len(fg_inds)
      if len(fg_inds) > 32:
        fg_inds = np.random.choice(fg_inds, size=(32))
        num_fg = 32
      bg_inds = np.where(curr_labels == 0)[0]
      bg_inds = np.random.choice(bg_inds, size=(num_fg))
      curr_inds = np.concatenate((fg_inds, bg_inds))
      curr_tois = \
        np.concatenate((0, self.anchors[curr_inds]), axis=1)


      batch_tois = np.concatenate((batch_tois, curr_tois), axis=0)
      batch_label = np.concatenate((batch_label, curr_labels), axis=0)

    top[1].reshape(*batch_tois.shape)
    top[2].reshape(*batch_label.shape)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_tois.astype(np.float32, copy=False)
    top[2].data[...] = batch_label.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
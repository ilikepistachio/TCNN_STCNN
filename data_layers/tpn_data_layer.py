'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.jhmdb import jhmdb
import numpy as np
#from utils.cython_bbox import bbox_overlaps

class DataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    self.dataset = jhmdb('train', [self._height, self._width], split=1)
    self.anchors = self.dataset.get_anchors()
    self.num_boxes = 32

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # Ground truth labels.
    top[1].reshape(self._batch_size * self.num_boxes)
    # Ground truth tois.
    top[2].reshape(self._batch_size * self.num_boxes, 5)

  def forward(self, bottom, top):
    [clips, labels, tmp_bboxes, box_idx] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_tois = np.empty((self._batch_size, 5))
    batch_labels = np.empty((self._batch_size))

    u_i = np.unique(box_idx)
    for i in u_i:
      curr_idx = np.where(box_idx == i)[0]
      box = tmp_bboxes[curr_idx, :, 1 : 5]
      gt_bboxes = np.array(np.mean(box, axis=1)) / 16

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      curr_labels = np.ones(self.anchors.shape[0]) * (-1)
      curr_labels[max_overlaps < 0.5] = 0
      curr_labels[max_overlaps >= 0.6] = labels[i]

      curr_labels[gt_argmax_overlaps] = labels[i]

      fg_inds = np.where(curr_labels > 0)[0]
      num_fg = len(fg_inds)
      if len(fg_inds) > 16:
        fg_inds = np.random.choice(fg_inds, size=(16))
        num_fg = 16

      bg_inds = np.where(curr_labels == 0)[0]
      num_bg = num_fg
      bg_inds = np.random.choice(bg_inds, size=(num_bg))
      inds = np.hstack((fg_inds, bg_inds))
      curr_bboxes = np.hstack((np.ones((len(inds), 1)) * i, self.anchors[inds]))
      batch_tois = np.vstack((batch_tois, curr_bboxes))
      batch_labels = np.hstack((batch_labels, curr_labels[inds]))
      i += 1


    top[1].reshape(*batch_labels.shape)
    top[2].reshape(*batch_tois.shape)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_labels.astype(np.float32, copy=False)
    top[2].data[...] = batch_tois.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
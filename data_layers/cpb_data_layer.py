'''
The Caffe data layer for training Coarse Proposal Boxes.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/3d-caffe/python')
import caffe
from dataset.ucf_sports import UcfSports
import numpy as np
from rpn.generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps

class CpbDataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self._batch_size = 6
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = UcfSports('train', [self._height, self._width],
                             '/home/rhou/ucf_sports')

    self._feat_stride = 16
    self._pooled_height = np.round(self._height / float(self._feat_stride))
    self._pooled_width = np.round(self._width / float(self._feat_stride))

    self._root_anchors = generate_anchors(ratios=[0.5, 1, 2, 4],
                                          scales=np.array([3, 6, 8, 11, 14]))
    self.num_anchors = self._root_anchors.shape[0]
    shift_x = np.arange(0, self._pooled_width) * self._feat_stride
    shift_y = np.arange(0, self._pooled_height) * self._feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    A = self.num_anchors
    all_anchors = (self._root_anchors.reshape((1, A, 4)).transpose((1, 0, 2)) +
                   shifts.reshape((1, K, 4)))
    all_anchors = all_anchors.reshape((K * A, 4))
    self.total_anchors = int(K * A)
    self.inds_inside = np.where(
      (all_anchors[:, 0] >= 0) &
      (all_anchors[:, 1] >= 0) &
      (all_anchors[:, 2] < self._width + 0) &  # width
      (all_anchors[:, 3] < self._height + 0)  # height
    )[0]
    self.anchors = all_anchors[self.inds_inside, :]
    self.len = len(self.inds_inside)

  @property
  def size(self):
    return len(self._vddb)

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # Ground truth bounding boxes.
    top[1].reshape(self._batch_size, 1, self.total_anchors)

  def forward(self, bottom, top):
    [batch_clip, _, batch_bboxes] = self.dataset.next_batch(self._batch_size,
                                                            self._depth,
                                                            random=True)
    batch_labels = np.empty((self._batch_size, 1, self.total_anchors))
    for i in xrange(self._batch_size):
      gt_boxes = batch_bboxes[i]
      labels = np.empty(self.len, dtype=np.float32)
      labels.fill(-1)

      corse_boxes = np.empty((gt_boxes.shape[0], 4))
      for j in xrange(gt_boxes.shape[0]):
        corse_boxes[j, 0] = gt_boxes[j, :, 0].min()
        corse_boxes[j, 1] = gt_boxes[j, :, 1].min()
        corse_boxes[j, 2] = gt_boxes[j, :, 2].max()
        corse_boxes[j, 3] = gt_boxes[j, :, 3].max()

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(corse_boxes, dtype=np.float))
      argmax_overlaps = overlaps.argmax(axis=1)
      max_overlaps = overlaps[np.arange(self.len), argmax_overlaps]
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      labels[max_overlaps < 0.3] = 0
      labels[max_overlaps >= 0.7] = 1
      labels[gt_argmax_overlaps] = 1

      num_fg = 32
      fg_inds = np.where(labels == 1)[0]
      if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
          fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

      # subsample negative labels if we have too many
      num_bg = 32 - np.sum(labels == 1)
      bg_inds = np.where(labels == 0)[0]
      if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(
          bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

      batch_labels[i] = _unmap(labels, self.total_anchors,
                               self.inds_inside, fill=-1).reshape(1, -1)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_labels.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret
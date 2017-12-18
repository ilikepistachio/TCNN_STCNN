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
    self._height = 240
    self._width = 320
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
    # diff
    top[3].reshape(self._batch_size * self.num_boxes, 4)
    # mask
    top[4].reshape(self._batch_size * self.num_boxes, 4)
    # toi2
    top[5].reshape(self._batch_size * self.num_boxes, 5)


  def forward(self, bottom, top):
    [clips, labels, tmp_bboxes, box_idx] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_tois = np.empty((0, 5))
    batch_label = np.empty((0, 1))
    batch_diff = np.empty((0, 4))
    batch_mask = np.empty((0, 4))
    batch_toi2 = np.empty((0, 5))

    for i in xrange(self._depth):
      box = tmp_bboxes[0, :, :]
      gt_bboxes = np.expand_dims((box[i] / 16), axis=0)

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      curr_labels = np.ones(self.anchors.shape[0]) * (-1)
      curr_labels[max_overlaps < 0.4] = 0
      curr_labels[max_overlaps >= 0.6] = 1
      curr_labels[gt_argmax_overlaps] = 1

      fg_inds = np.where(curr_labels > 0)[0]
      num_fg = len(fg_inds)
      if len(fg_inds) > 4:
        fg_inds = np.random.choice(fg_inds, size=(4))
        num_fg = 4
      bg_inds = np.where(curr_labels == 0)[0]
      bg_inds = np.random.choice(bg_inds, size=(num_fg))
      curr_inds = np.concatenate((fg_inds, bg_inds))
      curr_i = np.ones((num_fg * 2, 1)) * i
      curr_tois = \
        np.concatenate((curr_i, self.anchors[curr_inds]), axis=1)
      curr_toi2 = np.concatenate((np.zeros((num_fg * 2, 1)),
                                  self.anchors[curr_inds]), axis=1)
      curr_l = np.expand_dims(curr_labels[curr_inds], axis=1)
      num_samples = 2 * num_fg
      fg_diff = bbox_transform(self.anchors[fg_inds], gt_bboxes)
      curr_diff = np.zeros((num_samples, 4))
      curr_diff[0: num_fg] = fg_diff
      curr_mask = np.repeat(curr_l, 4, axis=1)

      batch_tois = np.concatenate((batch_tois, curr_tois), axis=0)
      batch_label = np.concatenate((batch_label, curr_l), axis=0)
      batch_diff = np.concatenate((batch_diff, curr_diff), axis=0)
      batch_mask = np.concatenate((batch_mask, curr_mask), axis=0)
      batch_toi2 = np.concatenate((batch_toi2, curr_toi2), axis=0)

    top[1].reshape(*batch_tois.shape)
    top[2].reshape(*batch_label.shape)
    top[3].reshape(*batch_diff.shape)
    top[4].reshape(*batch_mask.shape)
    top[5].reshape(*batch_toi2.shape)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_tois.astype(np.float32, copy=False)
    top[2].data[...] = batch_label.astype(np.float32, copy=False)
    top[3].data[...] = batch_diff.astype(np.float32, copy=False)
    top[4].data[...] = batch_mask.astype(np.float32, copy=False)
    top[5].data[...] = batch_toi2.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
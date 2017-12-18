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

class DataEval():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    self.dataset = jhmdb('train', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    [clips, gt_bboxes, gt_label, vid_name, is_last] \
      = self.dataset.next_val_video()


    num_frames = clips.shape[0]
    for i in xrange(num_frames - self._depth + 1):
      batch_clip = np.expand_dims(clips[i : i + self._depth].transpose(3, 0, 1, 2), axis=0)
      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)

      self._net.forward()
      result = self._net.blobs['loss'].data[0, 1]
      result = result.reshape(self._anchor_dims).transpose((1, 2, 0))
      curr_gt = np.mean(gt_bboxes[i : i + self._depth, 1 : 5], axis=0) / 16
      curr_gt = np.expand_dims(curr_gt, axis=0)
      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(curr_gt, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      curr_labels = np.ones(self._anchor_dims[0] *
                            self._anchor_dims[1] *
                            self._anchor_dims[2]) * (-1)
      curr_labels[self.valid_idx[max_overlaps < 0.5]] = 0
      curr_labels[self.valid_idx[max_overlaps > 0.6]] = 1
      curr_labels[self.valid_idx[gt_argmax_overlaps]] = 1
      curr_labels = curr_labels.reshape(
        (self._anchor_dims[1], self._anchor_dims[2], self._anchor_dims[0]))

      pass

    '''
    for i in u_i:
      curr_idx = np.where(box_idx == i)[0]
      box = tmp_bboxes[curr_idx, :, :]
      gt_bboxes = np.mean(box, axis=1) / 16

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
      batch_labels[i] = curr_labels.reshape((-1, 2)).transpose((1, 0))

      
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
      '''


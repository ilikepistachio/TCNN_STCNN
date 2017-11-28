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
from utils.bbox_transform import bbox_transform_inv

class RegDataLayer():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = UcfSports('test'
                             '', [self._height, self._width],
                             '/home/rhou/ucf_sports')

    self.anchors = self.dataset.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    [clips, labels, tmp_bboxes, _] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_tois = np.empty((0, 5))

    i = 0
    for box in tmp_bboxes:
      gt_bboxes = np.mean(box, axis=1) / 16

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)
      argmax_overlaps = overlaps.argmax(axis=1)

      curr_labels = np.ones(self.anchors.shape[0]) * (-1)
      curr_labels[max_overlaps < 0.3] = 0
      curr_labels[max_overlaps >= 0.7] = labels[i]

      curr_labels[gt_argmax_overlaps] = labels[i]

      fg_inds = np.where(curr_labels > 0)[0]
      num_fg = len(fg_inds)
      if len(fg_inds) > 16:
        fg_inds = np.random.choice(fg_inds, size=(16))
        num_fg = 16

      curr_bboxes = np.hstack((np.ones((num_fg, 1)) * i,
                               self.anchors[fg_inds]))

      batch_tois = np.vstack((batch_tois, curr_bboxes))
      i += 1

    self._net.blobs['data'].reshape(self._batch_size, 3, self._depth,
                                    self._height, self._width)
    self._net.blobs['tois'].reshape(*batch_tois.shape)

    self._net.blobs['data'].data[...] = batch_clip.astype(np.float32, copy=False)
    self._net.blobs['tois'].data[...] = batch_tois.astype(np.float32, copy=False)

    self._net.forward()
    result = self._net.blobs['fc8-2'].data[...]
    print(result.shape)
    pred = _unmap(curr_labels[fg_inds], result, batch_tois[:, 1 : 5])
    print(pred * 16)
    print(gt_bboxes * 16)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

def _unmap(label, target, tois):
  l = label.size
  r_diff = np.zeros((l, 4))
  for i in xrange(l):
    curr_label = int(label[i] - 1)
    r_diff[i] = target[i, curr_label * 4 : curr_label * 4 + 4]

  pred = bbox_transform_inv(tois, r_diff)
  return pred
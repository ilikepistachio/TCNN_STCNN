'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
from dataset.jhmdb import jhmdb
import numpy as np
from utils.cython_bbox import bbox_overlaps
from utils.bbox_transform import bbox_transform


class DataEval():
  def __init__(self):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    self.dataset = jhmdb('trainval', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()

  def forward(self):
    [clips, gt_bboxes, gt_label, vid_name, is_last] \
      = self.dataset.next_val_video()

    num_frames = clips.shape[0]
    r1 = []
    r2 = []
    for i in xrange(num_frames - self._depth + 1):
      curr_gt = np.mean(gt_bboxes[i: i + self._depth, 1: 5], axis=0) / 16
      curr_gt = np.expand_dims(curr_gt, axis=0)
      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(curr_gt, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)
      gt_max_overlaps = overlaps.max(axis=0)

      curr_labels = np.ones(self._anchor_dims[0] *
                            self._anchor_dims[1] *
                            self._anchor_dims[2]) * (-1)
      curr_labels[self.valid_idx[max_overlaps < 0.5]] = 0
      curr_labels[self.valid_idx[max_overlaps > 0.6]] = 1
      curr_labels[self.valid_idx[gt_argmax_overlaps]] = 1
      l = max_overlaps > 0.6
      l[gt_argmax_overlaps] = True
      ol = overlaps[l]
      pos_box = self.anchors[l]
      diff = bbox_transform(pos_box, curr_gt)
      r1.append(gt_max_overlaps)
      r2.append(np.abs(diff).max(axis=0))

    return r1, r2, is_last

if __name__ == '__main__':
  c = DataEval()
  t1 = []
  t2 = []
  while(1):
    [r1, r2, flag] = c.forward()
    t1.append(np.array(r1).min())
    t2.append(np.array(r2).max(axis=0))
    if (flag):
      break

  t1 = np.array(t1)
  t2 = np.array(t2)
  pass





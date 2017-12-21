'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.jhmdb import jhmdb
import numpy as np
import cv2
import os
from utils.bbox_transform import bbox_transform_inv
from utils.cython_bbox import bbox_overlaps

class DataEval():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    self.dataset = jhmdb('val', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    [clips, gt_bboxes, gt_label, vid_name, is_last] \
      = self.dataset.next_val_video()

    if not(os.path.isdir('data/jhmdb/tpn/{}'.format(vid_name))):
      os.makedirs('data/jhmdb/tpn/{}'.format(vid_name))
    num_frames = clips.shape[0]
    scores = np.zeros((num_frames, 3600))
    weights = np.zeros(num_frames)
    diff = np.zeros((num_frames, 4, 3600))
    for i in xrange(num_frames - self._depth + 1):
      batch_clip = np.expand_dims(clips[i : i + self._depth].transpose(3, 0, 1, 2), axis=0)
      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      self._net.forward()
      s = self._net.blobs['loss'].data[:, 1]
      d = self._net.blobs['reg_score'].data[...]
      scores[i: i + self._depth] += s
      weights[i : i + self._depth] += np.ones(self._depth)
      diff[i : i + self._depth] += d

    curr_score = scores[0].reshape((12,15,20)).transpose((1,2,0)).reshape(-1)[self.valid_idx]
    curr_diff = \
      diff[0].reshape((4, 12, 15, 20)).transpose((2, 3, 1, 0)).reshape((-1, 4))[
        self.valid_idx]
    selected_idx = np.argsort(curr_score)[-40:]
    prev_s = curr_score[selected_idx]
    prev_pred = pred_bbox(self.anchors[selected_idx], curr_diff[selected_idx])
    prev_state = []
    preds = []
    preds.append(prev_pred)
    for j in xrange(40):
      prev_state.append((j,))

    for i in xrange(1, num_frames):
      curr_score = scores[i].reshape((12,15,20)).transpose((1,2,0)).reshape(-1)[self.valid_idx]
      curr_diff = diff[i].reshape((4, 12, 15, 20)).transpose((2, 3, 1, 0)).reshape((-1, 4))[self.valid_idx] / weights[i]

      selected_idx = np.argsort(curr_score)[-40 :]
      curr_s = curr_score[selected_idx]
      curr_pred = pred_bbox(self.anchors[selected_idx], curr_diff[selected_idx])
      preds.append(curr_pred)

      overlaps = bbox_overlaps(
        np.ascontiguousarray(curr_pred, dtype=np.float),
        np.ascontiguousarray(prev_pred, dtype=np.float)) + prev_s
      idx = overlaps.argmax(axis=1)
      curr_state = []
      for j in xrange(40):
        curr_s[j] += overlaps[j, idx[j]]
        curr_state.append(prev_state[idx[j]] + (j,))

      prev_s = curr_s
      prev_pred = curr_pred
      prev_state = curr_state

    selected = prev_state[curr_s.argmax()]
    det = np.empty((num_frames, 4))
    for i in xrange(num_frames):
      det[i] = preds[i][selected[i]] * 16

      '''
      curr = (preds[i][selected[i]] * 16).astype(np.uint32)
      img = cv2.imread(
        '/home/rhou/JHMDB/Rename_Images/{}/{:05d}.png'.format(vid_name, i + 1))
      cv2.rectangle(img, (curr[0], curr[1]), (curr[2], curr[3]),
                    color=(0, 0, 255))
      cv2.imwrite('/home/rhou/tmp/{}/{:05d}.png'.format(vid_name, i + 1), img)
      '''

    np.save('data/jhmdb/tpn/{}/bboxes.npy'.format(vid_name), det)
    return is_last

def pred_bbox(anchors, diff):
  for j in xrange(40):
    diff[j, 0] = max(min(diff[j, 0], 0.3), -0.3)
    diff[j, 1] = max(min(diff[j, 1], 0.3), -0.3)
    diff[j, 2] = max(min(diff[j, 2], 0.5), -0.5)
    diff[j, 3] = max(min(diff[j, 3], 0.5), -0.5)
  pred = bbox_transform_inv(anchors, diff)
  for j in xrange(40):
    pred[j, 0] = max(min(pred[j, 0], 19), 0)
    pred[j, 1] = max(min(pred[j, 1], 14), 0)
    pred[j, 2] = max(min(pred[j, 2], 19), 0)
    pred[j, 3] = max(min(pred[j, 3], 14), 0)

  return pred

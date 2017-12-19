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

class DataEval():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    self.dataset = jhmdb('eval', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    [clips, gt_bboxes, gt_label, vid_name, is_last] \
      = self.dataset.next_val_video()

    os.makedirs('/home/rhou/tmp/{}'.format(vid_name))
    num_frames = clips.shape[0]
    scores = np.zeros((num_frames, 3600))
    for i in xrange(num_frames - self._depth + 1):
      batch_clip = np.expand_dims(clips[i : i + self._depth].transpose(3, 0, 1, 2), axis=0)
      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      self._net.forward()
      result = self._net.blobs['loss'].data[:, 1]
      scores[i: i + self._depth] = result

    for i in xrange(num_frames):
      curr = scores[i].reshape((12,15,20)).transpose((1,2,0)).reshape(-1)[self.valid_idx]
      p = curr.argmax()
      pred = (self.anchors[p] * 16).astype(np.uint32)
      img = cv2.imread('/home/rhou/JHMDB/Rename_Images/{}/{:05d}.png'.format(vid_name, i + 1))
      cv2.rectangle(img, (pred[0], pred[1]), (pred[2], pred[3]), color=(0,0,255))
      cv2.imwrite('/home/rhou/tmp/{}/{:05d}.png'.format(vid_name, i + 1), img)

    return is_last


'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.thumos14 import thumos14
import numpy as np
from utils.cython_bbox import bbox_overlaps

class RecDataLayer():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self._num_anchors = 10
    self.dataset = thumos14('train', [self._height, self._width],
                             '/home/rhou/thumos14')
    self.base_anchors = np.load('/home/rhou/thumos14/cache/anchors_{}_{}.npy'.format(self._depth, self._num_anchors)).transpose()

    self._bottom_height = np.ceil(self._height / 16.0)
    self._bottom_width = np.ceil(self._width / 16.0)
    self._bottom_width = np.ceil(self._width / 16.0)
    shift_x = np.arange(0, self._bottom_width)
    shift_y = np.arange(0, self._bottom_height)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = self._num_anchors
    K = shifts.shape[0]
    all_anchors = (self.base_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where(
      (all_anchors[:, 0] >= 0) &
      (all_anchors[:, 1] >= 0) &
      (all_anchors[:, 2] < self._bottom_width) &  # width
      (all_anchors[:, 3] < self._bottom_width)  # height
    )[0]

    self.anchors = all_anchors[inds_inside, :]

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)
    self._list = []

  def forward(self):
    [clips, labels, gt_bboxes, gt_length, is_last] \
      = self.dataset.next_val_video(random=False)

    # Select negative clip.
    inds = []
    aa = gt_bboxes[0, :, 0]
    for i in xrange(clips.shape[0] - self._depth + 1):
      flag = False
      for j in xrange(i, i + self._depth):
        if np.where(aa == j)[0].size > 0:
          flag = True
      if not(flag):
        inds.append(i)

    result = np.empty((0, 3))
    for id in inds:
      batch_clip = np.expand_dims(clips[id : id + self._depth], axis=0)
      batch_clip = batch_clip.transpose([0, 4, 1, 2, 3])

      l = self.anchors.shape[0]
      curr_result = np.empty((0, 2))

      self._net.blobs['data'].reshape(self._batch_size, 3, self._depth,
                                      self._height, self._width)
      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      self._net.blobs['tois'].reshape(2190, 5)

      curr_anchors = np.hstack((np.zeros((self.anchors.shape[0], 1)),
                                self.anchors))

      self._net.blobs['tois'].data[...] \
        = curr_anchors.astype(np.float32, copy=False)
      self._net.forward()
      pred = self._net.blobs['prob'].data[...].argmax(axis=1)
      max_val = self._net.blobs['prob'].data[...].max(axis=1)
      pos_idx = np.where(pred > 0)[0]
      if pos_idx.size > 0:
        tmp = np.vstack((pos_idx, max_val[pos_idx]))
        curr_result = np.vstack((curr_result, tmp.transpose()))

      result = np.vstack((result,
                          np.hstack((np.ones((curr_result.shape[0], 1)) * id,
                                     curr_result))))

    self._list.append(result)
    return is_last

  @property
  def show_list(self):
    return self._list
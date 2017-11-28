'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.thumos14 import thumos14
import numpy as np

class RecDataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self._batch_size = 2
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = thumos14('train', [self._height, self._width],
                             '/home/rhou/thumos14', anchors=True)

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # Ground truth labels.
    top[1].reshape(self._batch_size * 16)
    # Ground truth tois.
    top[2].reshape(self._batch_size * 16, 5)

  def forward(self, bottom, top):
    [clips, batch_labels, batch_tois, _] \
      = self.dataset.next_adv_batch()
    batch_clip = clips.transpose((0, 4, 1, 2, 3))

    top[1].reshape(*batch_labels.shape)
    top[2].reshape(*batch_tois.shape)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_labels.astype(np.float32, copy=False)
    top[2].data[...] = batch_tois.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
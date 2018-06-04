'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from datasets.davis import Davis
import numpy as np

class FakeData():
  def __init__(self,height, width):
    self.height = height
    self.width = width
    import cv2
    import glob
    files = glob.glob('images/*.jpg')
    self.images = np.array([cv2.imread(f) for f in files])
    self.annotation = np.array([cv2.imread(f[:-4] + '.png') for f in files])

  def next_batch(self, batch_size, depth):
    images = np.empty(batch_size, depth, self.height, self.width, 3)
    labels = np.empty(batch_size, depth, self.height, self.width, 1)
    for i in xrange(batch_size):
      begin_idx = np.random.randint(self.images.shape[0] - depth + 1)
      images[i] = self.images[begin_idx:begin_idx + depth, :, :, :]
      labels[i] = self.annotation[begin_idx:begin_idx + depth, :, :, :]

    return images, labels


class DataLayer(caffe.Layer):
  def setup(self, bottom, top):

    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320
    if (True):
      self.dataset = FakeData(self._height, self._width)
    else:
      self.dataset = Davis('train', '2016', [self._height, self._width],
                           '/home/mubarakshah/DAVIS')

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 3, self._depth, self._height, self._width)
    # Ground truth labels.
    top[1].reshape(1, 1, self._depth, self._height, self._width)


  def forward(self, bottom, top):
    [clips, _, batch_seg] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_seg = np.expand_dims(batch_seg, axis=1)

    batch_seg = batch_seg > 0
    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_seg.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

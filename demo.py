'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rui/caffe/python')
import caffe
import numpy as np
from os import mkdir
from os.path import exists, join
import cv2
import matplotlib.pyplot as plt

class DataLayer():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 240
    self._width = 320

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)
    self._net.blobs['data'].reshape(1, 3, 8, self._height, self._width)

    self.images = self.load_images()

  def load_images(self):
    import glob
    files = glob.glob('images/*.png')
    return np.array([cv2.resize(cv2.imread(f), (320, 240)) for f in files])

  def show(self):
    num_frames = self.images.shape[0]
    num_clips = num_frames // self._depth

    for i in range(num_clips):
      curr_clip = self.images[i * self._depth:i * self._depth + self._depth]
      batch_clip = curr_clip.transpose((3, 0, 1, 2))

      self._net.blobs['data'].data[0] = batch_clip
      self._net.forward()

      curr_prob = self._net.blobs['prob'].data[0, 1, :, :, :]
      curr_prediction = curr_prob > 0.4 * 255
      

if __name__ == '__main__':
  net = 'models/davis/c3d_deconv_test.prototxt'

  model = 'davis_deconv_d7_skip_pool_iter_5000.caffemodel'
  d = DataLayer(net, model)
  d.show()

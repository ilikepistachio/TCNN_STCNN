import sys
sys.path.insert(0, '/home/rhou/caffe/python')
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
    files = glob.glob('images/*.jpg')
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
      curr_prediction = curr_prob > 0.32
      plt.subplot(241)
      plt.imshow(curr_prediction[0] * 1.0)
      plt.subplot(242)
      plt.imshow(curr_prediction[1] * 1.0)
      plt.subplot(243)
      plt.imshow(curr_prediction[2] * 1.0)
      plt.subplot(244)
      plt.imshow(curr_prediction[3] * 1.0)
      plt.subplot(245)
      plt.imshow(curr_prediction[4] * 1.0)
      plt.subplot(246)
      plt.imshow(curr_prediction[5] * 1.0)
      plt.subplot(247)
      plt.imshow(curr_prediction[6] * 1.0)
      plt.subplot(248)
      plt.imshow(curr_prediction[7] * 1.0)
      plt.show()


if __name__ == '__main__':
  net = 'models/davis/c3d_deconv_test.prototxt'
  if len(sys.argv) > 1:
    model = sys.argv[1]
  else:
    import urllib
    urllib.urlretrieve ("http://www.cs.ucf.edu/~rhou/files/davis_240_320.caffemodel", "davis_240_320.caffemodel")
    model = 'davis_240_320.caffemodel'
  d = DataLayer(net, model)
  d.show()
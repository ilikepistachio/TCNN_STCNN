'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from skimage.io import imread
import numpy as np
import os

class DataLayer():
  def __init__(self, net, weights):
    self._batch_size = 1
    self._depth = 16
    self._split = 1
    with open('/home/rhou/ucf101_deep/src/video_list.txt') as f:
      lines = f.readlines()

    self.sequence = []

    for line in lines:
      line = line.strip()
      p0 = line.find(' ')
      p2 = line[: p0].rfind('_g')
      group = int(line[p2 + 2 : p2 + 4])
      if not((group - 1) / 7 == (self._split - 1)):
        continue
      p1 = line.rfind(' ')
      video_name = line[: p0]
      label = int(line[p0 + 1 : p1])
      frames = int(line[p1 + 1 :])

      self.sequence.append({'name': video_name,
                            'label': label,
                            'frame': frames})

    print('Totally %d samples.' % len(self.sequence))

    self._mean = np.load('/home/rhou/ucf101_deep/mean.npy')

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, weights, caffe.TEST)
    self.pos = 0

  @property
  def seq_size(self):
    return len(self.sequence)

  @property
  def pos_num(self):
    return self.pos

  def forward(self, idx):
    curr_seq = self.sequence[idx]

    [batch_clip, batch_label] = _load_frames(curr_seq)
    n = int(np.floor(batch_clip.shape[0] / 16.0))
    batch_clip = batch_clip[: 16 * n].reshape((n, 16, 128, 171, 3))
    batch_clip = batch_clip.transpose([0, 4, 1, 2, 3]) - self._mean

    batch_clip = batch_clip[:, :, :, 8 : 120, 30 : 142]

    self._net.blobs['data'].reshape(n, 3, 16, 112, 112)
    self._net.blobs['data'].data[...] = batch_clip.astype(np.float32, copy=False)

    self._net.forward()

    r = self._net.blobs['prob'].data[...].argmax()
    if r == batch_label:
      self.pos += 1

def _load_frames(video):
  frames = video['frame']
  ims = np.empty((frames, 128, 171, 3), dtype=np.float32)
  for i in xrange(frames):
    im_name = os.path.join('/home/rhou/ucf101_deep/frames',
                                 video['name'], '%07d.png' % (i + 1))
    im = imread(im_name)
    ims[i] = np.array(im, dtype=np.float32)[:,:,::-1]

  return ims, np.array(video['label'])

if __name__ == '__main__':
  net = '/home/rhou/videoflow/models/c3d_test.prototxt'
  model = '/home/rhou/models/ucf101_c3d_v2_iter_5000.caffemodel'
  r = DataLayer(net, model)
  n = r.seq_size
  for i in xrange(n):
    r.forward(i)

  print(r.pos_num)
  print(r.pos_num / float(n))
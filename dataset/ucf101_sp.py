'''
The Caffe data layer for training Coarse Proposal Boxes.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
import numpy as np
import cv2
import os

class DataLayer(caffe.Layer):
  def setup(self, bottom, top):
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
      if (group - 1) % 7 == (self._split - 1):
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
    self._curr = 0

  def reshape(self, bottom, top):
    top[0].reshape(1, 3, self._depth, 112, 112)
    top[1].reshape(1)

  def forward(self, bottom, top):
    if self._curr == self._len:
      self._curr = 0
    if self._curr == 0:
      np.random.shuffle(self.sequence)
    [batch_clip, batch_label] = _load_frames(self.sequence[self._curr])
    batch_clip = batch_clip - self._mean
    batch_clip = _random_crop(batch_clip)
    self._curr += 1

    top[0].reshape(*batch_clip.shape)
    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_label.astype(np.float32, copy=False)

  def backward(self, bottom, top):
    pass

def _load_frames(video):
  frames = video['frame']
  ims = np.empty((frames, 128, 171, 3), dtype=np.float32)
  for i in xrange(frames):
    im_name = os.path.join('/home/rhou/ucf101_deep/frames',
                                 video['name'], '%07d.png' % (i + 1))
    print(im_name)
    im = cv2.imread(im_name)
    print(i)
    ims[i] = np.array(im, dtype=np.float32)
  n = np.floor(ims.shape[0] / 16.0).astype(dtype=np.int32)
  ims = ims[: 16 * n].reshape((n, 16, 128, 171, 3)).transpose((0, 4, 1, 2, 3))
  return ims, np.array(video['label'])

def _random_crop(im_data):
  w = np.random.randint(128 - 112 + 1)
  h = np.random.randint(171 - 112 + 1)
  is_flip = np.random.randint(2)
  im_data = im_data[:, :, :, w : w + 112, h : h + 112]
  if is_flip:
    result = im_data[:, :, :, :, : : -1]
  else:
    result = im_data
  return result

if __name__ == "__main__":
  d = DataLayer()
  d.forward()
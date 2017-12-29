from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import cv2
import numpy as np
import os.path
import scipy.io as sio
import glob

class abnormal():
  def __init__(self, name, clip_shape, split=1):
    self._name = name
    self._data_path = '/home/rhou/UCF_AnomalyDataset'
    self._vddb = []
    self._height = clip_shape[0]
    self._width = clip_shape[1]
    self._split = split - 1

    self._num_classes = 15
    self._classes = ('Normal_Videos',  # always index 0
                     'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
                     'Explosion', 'Fighting', 'Normal_Videos_event',
                     'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
                     'Stealing', 'Vandalism')
    self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
    cache_file = os.path.join(self._data_path, 'cache',
        'abnormal_%d_%d_db.pkl' % (self._height, self._width))
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        self._vddb = cPickle.load(fid)
      print ('{} gt vddb loaded from {}'.format(self._name, cache_file))
    else:
      self._vddb = self._read_video_list()

      with open(cache_file, 'wb') as fid:
        cPickle.dump(self._vddb, fid, cPickle.HIGHEST_PROTOCOL)

    self._curr_idx = 0

    mean_file = os.path.join(self._data_path, 'cache',
                             'mean_frame_{}_{}.npy'.format(self._height,
                                                           self._width))
    if os.path.exists(mean_file):
      self._mean_frame = np.load(mean_file)
    else:
      self._mean_frame = self.compute_mean_frame()

    if name == 'train':
      self._vddb = self.keeps(1)
    else:
      if name == 'val':
        self._vddb = self.keeps(0)

  @property
  def vddb(self):
    return self._vddb
  @property
  def size(self):
    return len(self._vddb)

  def keeps(self, num):
    result = []
    for i in xrange(len(self.vddb)):
      if self.vddb[i]['split'][self._split] == num:
        result.append(self.vddb[i])
    return result

  def _read_video_list(self):
    vddb = []
    tmp = []
    file_name1 = os.path.join(self._data_path, 'splits',
                              'train_{:03d}.txt'.format(1))
    file_name2 = os.path.join(self._data_path, 'splits',
                              'test_{:03d}.txt'.format(1))
    if not os.path.isfile(file_name1):
      raise NameError('The video list file does not exists: ' + file_name1)
    with open(file_name1) as f:
      lines = f.readlines()
    num_train = len(lines)
    with open(file_name2) as f:
      lines = lines + f.readlines()

    i = 0
    for line in lines:
      split = np.zeros(4, dtype=np.uint8)
      video_name = line.strip()[:-4]
      label = self._class_to_ind[line[: line.find('/')]]
      print(video_name)
      if label == 0:
        continue
      if i < num_train:
        split[0] = 1
      _, video = self.clip_reader(video_name)
      vddb.append({'video_name': video_name,
                   'split': split,
                   'label': label,
                   'video': video})
      tmp.append(video_name)
      i = i + 1

    for split in xrange(1,4):
      file_name1 = os.path.join(self._data_path, 'splits',
                                'train_{:03d}.txt'.format(split + 1))
      file_name2 = os.path.join(self._data_path, 'splits',
                                'test_{:03d}.txt'.format(split + 1))
      with open(file_name1) as f:
        lines = f.readlines()
      num_train = len(lines)
      with open(file_name2) as f:
        lines = lines + f.readlines()

      i = 0
      for line in lines:
        video_name = line.strip()[:-4]
        if line[: line.find('/')] == 'Normal_Videos':
          continue
        index = tmp.index(video_name)
        if i < num_train:
          vddb[index]['split'][split] = 1
        i = i + 1

    return vddb

  def clip_reader(self, video_prefix):
    """Load frames in the clip.

    Using openCV to load the clip frame by frame.
    If specify the cropped size (crop_size > 0), randomly crop the clip.

      Args:
        index: Index of a video in the dataset.

      Returns:
        clip: A matrix (channel x depth x height x width) saves the pixels.
      """
    clip = np.empty((16, 8, self._height, self._width, 3))
    r1 = 0
    framepath = os.path.join(self._data_path, 'frame', video_prefix)
    unit = (len(glob.glob(framepath + '/*.png')) - 8) / 16
    for i in xrange(0, 16):
      for j in xrange(8):
        filename = os.path.join(
            self._data_path, 'frame', video_prefix,
            '%08d.png' % (i * unit + j + 1))

        im = cv2.imread(filename)
        if r1 == 0:
          r1 = self._height / im.shape[0]
          r2 = self._width / im.shape[1]
        im = cv2.resize(im, None, None, fx=r2, fy=r1,
                        interpolation=cv2.INTER_LINEAR)
        clip[i, j] = im
    return [r1, r2], np.asarray(clip, dtype=np.uint8)

  def next_batch(self, batch_size, depth):
    """Load next batch to feed the network.

      Args:
        batch_size: Number of examples per batch.
        depth: Clip length of clips.

      Return:
        batch_video: 5D tensor (batch_size x depth x height x width x channel).
                     Pixel information of all clips.
        batch_label: 1D tensor (batch_size). Ground truth label of samples.
        batch_bboxes: 2D tensor. Ground truth bounding boxes.
        is_last: If it is the last batch (used for eval).
    """
    batch_video = np.empty((batch_size, depth, self._height, self._width, 3))
    batch_label = np.empty(batch_size)
    batch_bboxes = np.empty((batch_size, depth, 4))
    batch_idx = np.arange(batch_size)
    is_last = False
    for i in xrange(batch_size):
      if self._curr_idx == self.size:
        self._curr_idx = 0
      if self._curr_idx == 0:
        np.random.shuffle(self._vddb)

      video = self.vddb[self._curr_idx]
      f_idx = np.random.randint(0, 16)
      tmp_video = video['video'][f_idx] - self._mean_frame

      if self._name == 'train' and np.random.randint(0, 2) == 1:
        tmp_video = tmp_video[:, :, :: -1, :]

      batch_video[i] = tmp_video
      batch_label[i] = video['gt_label']
      tmp = np.array([[0,0,319,239]])
      batch_bboxes[i] = np.repeat(tmp, 8, axis=0)
      self._curr_idx += 1

    return batch_video, batch_label, batch_bboxes, batch_idx

  def next_val_video(self):
    video = self._vddb[self._curr_idx]['video'] - self._mean_frame
    tmp = np.array([[0,0,319,239]])
    gt_bboxes = np.repeat(tmp, video.shape[0], axis=0)
    gt_label = self._vddb[self._curr_idx]['gt_label']
    vid_name = self._vddb[self._curr_idx]['video_name']
    print(self._curr_idx)
    self._curr_idx += 1
    return video, \
           gt_bboxes, \
           gt_label, \
           vid_name, \
           self._curr_idx == self.size

  def compute_mean_frame(self):
    sum_frame = np.zeros((self._height, self._width, 3), dtype=np.float32)
    num_frames = 0
    for db in self._vddb:
      curr_frame = np.sum(db['video'], dtype=np.float32, axis=0)
      curr_frame = np.sum(curr_frame, axis=0)
      sum_frame += curr_frame
      num_frames += db['video'].shape[0] * db['video'].shape[1]
    sum_frame = sum_frame / num_frames
    np.save(os.path.join(self._data_path, 'cache',
                         'mean_frame_{}_{}.npy'.format(self._height,
                                                       self._width)),
            sum_frame)
    return sum_frame

if __name__ == '__main__':
  d = abnormal('train', [240, 320], split=1)
  d.next_batch(2, 8)

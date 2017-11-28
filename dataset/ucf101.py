from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import cv2
import numpy as np
import os.path
import h5py

class ucf101():
  def __init__(self, name, clip_shape, data_path):
    self._name = name
    self._vddb = []
    self._data_path = data_path
    self._height = clip_shape[0]
    self._width = clip_shape[1]
    self._num_classes = 25
    self._classes = ('__background__',  # always index 0
        'BasketballDunk', 'Basketball', 'Biking', 'CliffDiving',
        'CricketBowling', 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing',
        'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
        'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
        'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking',
        'WalkingWithDog')
    self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
    cache_file = os.path.join(self._data_path, 'cache',
                              self._name + '%d_%d_db.pkl' % (self._height,
                                                             self._width))
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        self._vddb = cPickle.load(fid)
      print('{} gt vddb loaded from {}'.format(self._name, cache_file))
    else:
      [video_prefix, num_frames] = self._read_video_list()

      self._vddb = [self._load_annotations(video_prefix[i], num_frames[i])
                    for i in xrange(len(num_frames))]

      with open(cache_file, 'wb') as fid:
        cPickle.dump(self._vddb, fid, cPickle.HIGHEST_PROTOCOL)

    self._curr_idx = 0

    # Load mean clip.
    mean_file = os.path.join(self._data_path, 'cache',
                             'mean_frame_{}_{}.npy'.format(self._height,
                                                           self._width))
    if os.path.exists(mean_file):
      self._mean_frame = np.load(mean_file)
    else:
      self._mean_frame = self.compute_mean_frame()

  @property
  def vddb(self):
    return self._vddb

  @property
  def size(self):
    return len(self._vddb)

  @property
  def num_classes(self):
    return self._num_classes

  def get_anchors(self):
    base_anchors = np.load(
      '/home/rhou/ucf101_deep/cache/anchors_8_15.npy').transpose()
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
    A = 15
    K = shifts.shape[0]
    all_anchors = (base_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    # only keep anchors inside the image
    inds_inside = np.where(
      (all_anchors[:, 0] >= 0) &
      (all_anchors[:, 1] >= 0) &
      (all_anchors[:, 2] < self._bottom_width) &  # width
      (all_anchors[:, 3] < self._bottom_width)  # height
    )[0]
    return all_anchors[inds_inside]

  def _load_annotations(self, video_prefix, num_frames):
    """Read video annotations from text files.
    Args:
      video_prefix: Prefix of video annotation files.
      num_frames: Frames of a video.
    Return:
      gt_labels: Ground-truth labels of bounding boxes.
      gt_bboxes: Ground-truth bounding boxes. Format[frame, w1, h1, w2, h2]
    """
    gt_file = os.path.join(self._data_path, 'gt',
                           video_prefix + '.avi', 'tubes.hdf5')
    print(video_prefix)
    if not os.path.isfile(gt_file):
      raise Exception(gt_file + 'does not exist.')
    f = h5py.File(gt_file, 'r')
    n = len(f)
    gt_bboxes = np.empty((0, 5))
    gt_length = np.empty((0))
    for i in xrange(n):
      data = f[str(i)][()]
      iii = np.where(np.logical_or(data[:,1] > 319, data[:,2] > 239))[0]
      if iii.size > 0:
        data = data[:iii[0]]
      l = data.shape[0]
      gt_bboxes = np.vstack((gt_bboxes, data))
      gt_length = np.hstack((gt_length, l - np.arange(l)))
    p1 = video_prefix.find('_')
    p2 = video_prefix.find('_', p1 + 1)
    cls = self._class_to_ind[video_prefix[p1 + 1 : p2]]
    gt_labels = cls

    frame_name = os.path.join(self._data_path, 'PNGImages', video_prefix,
        '%07d.png' % (1))
    if not os.path.isfile(frame_name):
      raise Exception(frame_name + 'does not exist.')
    im = cv2.imread(frame_name)
    ratio = [self._height / im.shape[0], self._width / im.shape[1]]

    gt_bboxes[:, 1] = gt_bboxes[:, 1] * ratio[1]
    gt_bboxes[:, 2] = gt_bboxes[:, 2] * ratio[0]
    gt_bboxes[:, 3] = gt_bboxes[:, 3] * ratio[1]
    gt_bboxes[:, 4] = gt_bboxes[:, 4] * ratio[0]

    idx = np.where(gt_bboxes[:, 1] < 0)[0]
    if idx.size > 0:
      gt_bboxes[idx, 1] = 0

    idx = np.where(gt_bboxes[:, 2] < 0)[0]
    if idx.size > 0:
      gt_bboxes[idx, 2] = 0

    idx = np.where(gt_bboxes[:, 3] > self._width - 1)[0]
    if idx.size > 0:
      gt_bboxes[idx, 3] = self._width - 1

    idx = np.where(gt_bboxes[:, 4] > self._height - 1)[0]
    if idx.size > 0:
      gt_bboxes[idx, 4] = self._height - 1

    return {'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'gt_length': gt_length,
            'num_frames': num_frames,
            'video_prefix': video_prefix,
            'video_scale': ratio}

  def _read_video_list(self):
    """Read ucf sports video list from a text file.

    Args:
      file_name: file which store the list of [video_file gt_label num_frames].
      clip_length: Number of frames in each clip.

    Returns:
      clip_db: A list save the [video_name, begin_idx, gt_label].
    """
    file_name = os.path.join(self._data_path, 'VideoSets', self._name + '.txt')
    if not os.path.isfile(file_name):
      raise NameError('The video list file does not exists: ' + file_name)
    with open(file_name) as f:
      lines = f.readlines()
    video_names = []
    frames = []
    for line in lines:
      p1 = line.find(' ')
      video_names.append(line[: p1])
      frames.append(int(line[p1 + 1:].strip()))
    return video_names, frames

  def clip_reader(self, video_prefix, begin_idx, end_idx, scale):
    """Load frames in the clip.

    Using openCV to load the clip frame by frame.
    If specify the cropped size (crop_size > 0), randomly crop the clip.

      Args:
        index: Index of a video in the dataset.

      Returns:
        clip: A matrix (channel x depth x height x width) saves the pixels.
    """
    clip = np.empty((end_idx - begin_idx, self._height, self._width, 3), dtype=np.uint8)
    for i in xrange(begin_idx, end_idx):
      filename = os.path.join(
          self._data_path, 'PNGImages', video_prefix,
          '%07d.png' % (i + 1))

      im = cv2.imread(filename)
      clip[i - begin_idx] = cv2.resize(im, None, None, fx=scale[1], fy=scale[0],
                                       interpolation=cv2.INTER_LINEAR)
    return clip

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
    batch_bboxes = []
    is_last = False
    for i in xrange(batch_size):
      if self._curr_idx == self.size:
        if self._name == 'train':
          self._curr_idx = 0
        else:
          batch_video = batch_video[: i]
          batch_label = batch_label[: i]
          is_last = True
          return batch_video, batch_label, batch_bboxes, is_last
      if self._curr_idx == 0:
        np.random.shuffle(self._vddb)

      video = self.vddb[self._curr_idx]
      print(video['video_prefix'])
      idx = np.where(video['gt_length'] >= depth)[0]
      curr_frame = int(np.random.choice(idx, size=(1)))
      f_idx = int(video['gt_bboxes'][curr_frame, 0])
      curr_idx = np.where(np.logical_and(video['gt_bboxes'][:,0] == f_idx, video['gt_length'] >= depth))[0]
      tmp_bbox = np.empty((0, depth, 4))

      for j in curr_idx:
        box = video['gt_bboxes'][j : j + depth, 1 : 5]
        tmp_bbox = np.vstack((tmp_bbox, np.expand_dims(box, axis=0)))

      # Load video pixels
      tmp_video = self.clip_reader(video['video_prefix'], f_idx, f_idx + depth,
                                   video['video_scale'])
      tmp_video = tmp_video - self._mean_frame

      if self._name == 'train' and np.random.randint(0, 2) == 1:
        tmp_video = tmp_video[:, :, :: -1, :]
        tmp_bbox = tmp_bbox[:, :, [2, 1, 0, 3]]
        tmp_bbox[:, :, [0, 2]] = self._width - tmp_bbox[:, :, [0, 2]]

      batch_video[i] = tmp_video
      batch_label[i] = video['gt_labels']
      batch_bboxes.append(tmp_bbox)
      self._curr_idx += 1

    return batch_video, batch_label, batch_bboxes, is_last

  def compute_mean_frame(self):
    sum_frame = np.zeros((self._height, self._width, 3), dtype=np.float32)
    num_frames = 0
    for db in self._vddb:
      print(db['video_prefix'])
      clip = self.clip_reader(db['video_prefix'], 0, db['num_frames'],
                              db['video_scale'])
      curr_frame = np.sum(clip, dtype=np.float32, axis=0)
      sum_frame += curr_frame
      num_frames += db['num_frames']
    sum_frame = sum_frame / num_frames
    np.save(os.path.join(self._data_path, 'cache',
                         'mean_frame_{}_{}.npy'.format(self._height,
                                                       self._width)),
            sum_frame)
    return sum_frame

  def cluster_bboxes(self, length=8):
    data = np.empty((0, 2))
    for db in self._vddb:
      #print(db['video_prefix'])
      boxes = db['gt_bboxes']
      idx = np.where(db['gt_length'] >= length)[0]
      for i in idx:
        curr = np.mean(boxes[i : i + length, 1 : 5], axis=0)
        x = (curr[2] - curr[0]) / 16
        y = (curr[3] - curr[1]) / 16

        data = np.vstack((data, np.array([x, y])))
        if x < 0 or y < 0:
          print(db['video_prefix'])
    import sklearn.cluster
    import matplotlib.pyplot as plt
    for anchors in xrange(4, 17):
      [centers, b, _] = sklearn.cluster.k_means(data, anchors)
      plt.figure(anchors)
      c = np.linspace(0, 1, anchors)
      for i in xrange(anchors):
        flag = b == i
        plt.plot(data[flag, 0], data[flag, 1], 'o', color=plt.cm.RdYlBu(c[i]))
        plt.xlabel('width')
        plt.ylabel('height')
      #plt.show()
      plt.savefig(os.path.join(self._data_path, 'anchors_{}_{}.png'.format(length, anchors)))
      cx1 = centers[:, 0] / 2
      cx2 = centers[:, 1] / 2
      r = np.vstack((-cx1, -cx2, cx1, cx2))
      np.save(os.path.join(self._data_path, 'cache', 'anchors_{}_{}.npy'.format(length, anchors)), r)

if __name__ == '__main__':
  dataset = ucf101('train', [300, 400], '/home/rhou/ucf101_deep')
  #a = dataset.get_anchors()
  print(a)

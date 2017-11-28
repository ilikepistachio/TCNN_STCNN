from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import cv2
import numpy as np
import os.path
from six.moves import xrange  # pylint: disable=redefined-builtin

class UcfSports():
  def __init__(self, name, clip_shape, data_path):
    self._name = name
    self._vddb = []
    self._data_path = data_path
    self._height = clip_shape[0]
    self._width = clip_shape[1]
    if name == 'train':
      self._is_train = True
    else:
      self._is_train = False
    self._num_classes = 11
    self._classes = ('__background__',  # always index 0
                     'Diving-Side', 'Golf-Swing', 'Kicking', 'Lifting',
                     'Riding-Horse', 'Run-Side', 'SkateBoarding-Front',
                     'Swing-Bench', 'Swing-SideAngle',
                     'Walk-Front')
    self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
    cache_file = os.path.join(self._data_path, 'cache',
                              self._name + '%d_%d_db.pkl' % (self._height,
                                                             self._width))
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        self._vddb = cPickle.load(fid)
      print ('{} gt vddb loaded from {}'.format(self._name, cache_file))
    else:
      [video_prefix, num_frames] = self._read_video_list()

      self._vddb = [self._load_annotations(video_prefix[i], num_frames[i])
                    for i in xrange(len(num_frames))]

      with open(cache_file, 'wb') as fid:
        cPickle.dump(self._vddb, fid, cPickle.HIGHEST_PROTOCOL)

    self._curr_idx = 0

    # Load mean clip.
    self._mean = np.load('/home/rhou/ucf101_deep/mean.npy').transpose(
      [1, 2, 3, 0])
    mean_file = os.path.join(self._data_path, 'cache',
                             'mean_frame_{}_{}.npy'.format(self._height,
                                                           self._width))
    if os.path.exists(mean_file):
      self._mean_frame = np.load(mean_file)
    else:
      self._mean_frame = self.compute_mean_frame()

  @property
  def num_classes(self):
    return self._num_classes
  @property
  def vddb(self):
    return self._vddb

  @property
  def size(self):
    return len(self._vddb)

  def get_anchors(self):
    base_anchors = np.load(
      self._data_path + '/cache/anchors_8_15.npy').transpose()
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
    print(video_prefix)
    gt_file = os.path.join(self._data_path, 'NPYannotation',
                           video_prefix + '.npy')
    if not os.path.isfile(gt_file):
      raise Exception(gt_file + 'does not exist.')
    gt_bboxes = np.load(gt_file)
    cls = self._class_to_ind[video_prefix[0: video_prefix.find("_")]]
    gt_labels = np.ones(gt_bboxes.shape[0]) * cls
    [ratio, video] = self.clip_reader(video_prefix, num_frames)
    gt_bboxes[:, :, 1] = gt_bboxes[:, :, 1] * ratio[1]
    gt_bboxes[:, :, 2] = gt_bboxes[:, :, 2] * ratio[0]
    gt_bboxes[:, :, 3] = gt_bboxes[:, :, 3] * ratio[1]
    gt_bboxes[:, :, 4] = gt_bboxes[:, :, 4] * ratio[0]
    return {'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'num_frames': num_frames,
            'video_prefix': video_prefix,
            'video_scale': ratio,
            'video': video}

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

  def clip_reader(self, video_prefix, num_frames):
    """Load frames in the clip.

    Using openCV to load the clip frame by frame.
    If specify the cropped size (crop_size > 0), randomly crop the clip.

      Args:
        index: Index of a video in the dataset.

      Returns:
        clip: A matrix (channel x depth x height x width) saves the pixels.
    """
    clip = []
    r1 = 0
    for i in xrange(num_frames):
      filename = os.path.join(
          self._data_path, 'JPEGImages',
          '%s_%03d.jpg' % (video_prefix, i + 1))

      im = cv2.imread(filename)
      if r1 == 0:
        r1 = self._height / im.shape[0]
        r2 = self._width / im.shape[1]
      im = cv2.resize(im, None, None, fx=r2, fy=r1,
                      interpolation=cv2.INTER_LINEAR)
      clip.append(im)
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
    batch_bboxes = []
    is_last = False
    for i in xrange(batch_size):
      if self._curr_idx == self.size:
        '''
        if self._name == 'train':
          self._curr_idx = 0
        else:
          batch_video = batch_video[: i]
          batch_label = batch_label[: i]
          is_last = True
          return batch_video, batch_label, batch_bboxes, is_last
        '''
        self._curr_idx = 0
      if self._curr_idx == 0:
        np.random.shuffle(self._vddb)
      video = self.vddb[self._curr_idx]
      total_frames = video['gt_bboxes'].shape[1]
      n = int(np.floor(total_frames / float(depth)))

      curr_frame = np.random.randint(0, n) * depth
      f_idx = int(video['gt_bboxes'][0, curr_frame, 0])
      tmp_video = video['video'][f_idx : f_idx + depth] - self._mean_frame
      tmp_bbox = video['gt_bboxes'][:, curr_frame : curr_frame + depth, 1 : 5]

      if self._name == 'train' and np.random.randint(0, 2) == 1:
        tmp_video = tmp_video[:, :, :: -1, :]
        tmp_bbox = tmp_bbox[:, :, [2, 1, 0, 3]]
        tmp_bbox[:, :, [0, 2]] = self._width - tmp_bbox[:, :, [0, 2]]

      batch_video[i] = tmp_video
      batch_label[i] = video['gt_labels'][0]
      batch_bboxes.append(tmp_bbox)
      self._curr_idx += 1

    batch_bboxes = np.array(batch_bboxes)
    return batch_video, batch_label, batch_bboxes, is_last

  def next_val_video(self, random=False):
    video = self.vddb[self._curr_idx]
    print(video['video_prefix'])
    self._curr_idx += 1
    if self._curr_idx == self.size:
      is_last = True
      self._curr_idx = 0
    else:
      is_last = False
    if self._curr_idx == 0 and random:
      np.random.shuffle(self._vddb)

    return video['video'] - self._mean_frame,\
           video['gt_labels'][0],\
           video['gt_bboxes'],\
           is_last

  def compute_mean_frame(self):
    sum_frame = np.zeros((self._height, self._width, 3), dtype=np.float32)
    num_frames = 0
    for db in self._vddb:
      curr_frame = np.sum(db['video'], dtype=np.float32, axis=0)
      sum_frame += curr_frame
      num_frames += db['video'].shape[0]
    sum_frame = sum_frame / num_frames
    np.save(os.path.join(self._data_path, 'cache',
                         'mean_frame_{}_{}.npy'.format(self._height,
                                                       self._width)),
            sum_frame)
    return sum_frame

  def cluster_bboxes(self, length=8, anchors=9):
    data = np.empty((0, 2))
    for db in self._vddb:
      boxes = db['gt_bboxes']
      l = boxes.shape[1] - length + 1
      for i in xrange(l):
        if not(boxes[0, i, 0] + length == boxes[0, i + length - 1, 0] + 1):
          print('Invalid boxes!')
          continue
        curr = np.mean(boxes[0, i : i + length, 1 : 5], axis=0)
        x = (curr[2] - curr[0]) / 16
        y = (curr[3] - curr[1]) / 16
        data = np.vstack((data, np.array([x, y])))
    import sklearn.cluster
    [centers, b, _] = sklearn.cluster.k_means(data, anchors)

    import matplotlib.pyplot as plt
    plt.figure(1)
    c = np.linspace(0, 1, anchors)
    for i in xrange(anchors):
      flag = b == i
      plt.plot(data[flag, 0], data[flag, 1], 'o', color=plt.cm.RdYlBu(c[i]))
      plt.xlabel('width')
      plt.ylabel('height')
    #plt.show()
    plt.savefig(os.path.join(self._data_path,
                             'anchors_{}_{}.png'.format(length, anchors)))
    cx1 = centers[:, 0] / 2
    cx2 = centers[:, 1] / 2
    r = np.vstack((-cx1, -cx2, cx1, cx2))
    np.save(os.path.join(self._data_path,
                         'cache',
                         'anchors_{}_{}.npy'.format(length, anchors)), r)

if __name__ == '__main__':
  dataset = UcfSports('train', [300, 400], '/home/rhou/ucf_sports/')
  for i in xrange(4, 16):
    dataset.cluster_bboxes(anchors=i)
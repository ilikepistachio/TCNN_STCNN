from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import cv2
import numpy as np
import os.path
import h5py
from utils.cython_bbox import bbox_overlaps

class thumos14():
  def __init__(self, name, clip_shape, data_path, anchors=False):
    self._name = name
    self._vddb = []
    self._data_path = data_path
    self._height = clip_shape[0]
    self._width = clip_shape[1]
    if name == 'train':
      self._is_train = True
    else:
      self._is_train = False
    self._num_classes = 3
    self._classes = ('__background__',  # always index 0
                     'GolfSwing', 'TennisSwing')
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

    if anchors:
      self.get_anchors()
      self.generate_neg_db()

  @property
  def vddb(self):
    return self._vddb

  @property
  def size(self):
    return len(self._vddb)

  def generate_neg_db(self):
    with open('/home/rhou/thumos14/cache/neg_mining.pkl', 'rb') as f:
      neg_list = cPickle.load(f)
    if not(len(neg_list) == len(self._vddb)):
      raise Exception("Negative list size does not match!")
    self._negdb = []
    for i in xrange(len(neg_list)):
      if neg_list[i].size > 0:
        self._negdb.append({
          'video': self._vddb[i]['video'],
          'frame': neg_list[i],
        })

  def get_anchors(self):
    base_anchors = np.load(
      '/home/rhou/thumos14/cache/anchors_8_10.npy').transpose()
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
    A = 10
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
    self.anchors = all_anchors[inds_inside]

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
      l = f[str(i)][()].shape[0]
      gt_bboxes = np.vstack((gt_bboxes, f[str(i)][()]))
      gt_length = np.hstack((gt_length, l - np.arange(l)))
    gt_bboxes = np.expand_dims(gt_bboxes, axis=0)
    p1 = video_prefix.find('_')
    p2 = video_prefix.find('_', p1 + 1)
    cls = self._class_to_ind[video_prefix[p1 + 1 : p2]]
    gt_labels = np.ones(gt_bboxes.shape[0]) * cls
    [ratio, video] = self.clip_reader(video_prefix, num_frames)
    gt_bboxes[:, :, 1] = gt_bboxes[:, :, 1] * ratio[1]
    gt_bboxes[:, :, 2] = gt_bboxes[:, :, 2] * ratio[0]
    gt_bboxes[:, :, 3] = gt_bboxes[:, :, 3] * ratio[1]
    gt_bboxes[:, :, 4] = gt_bboxes[:, :, 4] * ratio[0]
    return {'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'gt_length': gt_length,
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
          self._data_path, 'PNGImages', video_prefix,
          '%07d.png' % (i + 1))

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
      total_frames = video['gt_bboxes'].shape[1]
      idx = np.where(video['gt_length'] >= depth)[0]
      curr_frame = np.random.choice(idx, size=(1))
      f_idx = int(video['gt_bboxes'][0, curr_frame, 0])

      # Read frames.
      tmp_video = video['video'][f_idx : f_idx + depth] - self._mean_frame
      tmp_bbox = video['gt_bboxes'][:, curr_frame : curr_frame + depth, 1 : 5]

      if self._name == 'train' and np.random.randint(0, 2) == 1:
        tmp_video = tmp_video[:, :, :: -1, :]
        tmp_bbox = tmp_bbox[:, :, [2, 1, 0, 3]]
        tmp_bbox[:, :, [0, 2]] = self._width - tmp_bbox[:, :, [0, 2]]

      batch_video[i] = tmp_video - self._mean_frame
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
           video['gt_length'],\
           is_last

  def next_adv_batch(self, depth=8):
    if not(depth == 8):
      raise Exception("Depth {} is not supported.".format(depth))
    batch_video = np.empty((2, depth, self._height, self._width, 3))
    batch_labels = np.empty((0))
    batch_tois = np.empty((0, 5))
    is_last = False
    if self._curr_idx == self.size:
      if self._name == 'train':
        self._curr_idx = 0
      else:
        is_last = True
    if self._curr_idx == 0 and self._name == 'train':
      np.random.shuffle(self._vddb)

    # Load positive video.
    video = self.vddb[self._curr_idx]
    total_frames = video['gt_bboxes'].shape[1]
    idx = np.where(video['gt_length'] >= depth)[0]
    curr_frame = int(np.random.choice(idx, size=(1)))
    f_idx = int(video['gt_bboxes'][0, curr_frame, 0])

    # Read frames.
    tmp_video = video['video'][f_idx: f_idx + depth] - self._mean_frame
    box = video['gt_bboxes'][0, curr_frame: curr_frame + depth, 1: 5]

    if self._name == 'train' and np.random.randint(0, 2) == 1:
      tmp_video = tmp_video[:, :, :: -1, :]
      box = box[:, [2, 1, 0, 3]]
      box[:, [0, 2]] = self._width - box[:, [0, 2]]

    batch_video[0] = tmp_video

    x1 = np.floor(np.mean(box[:, 0])) / 16  # , axis=1))
    y1 = np.floor(np.mean(box[:, 1])) / 16  # , axis=1))
    x2 = np.ceil(np.mean(box[:, 2])) / 16  # , axis=1))
    y2 = np.ceil(np.mean(box[:, 3])) / 16  # , axis=1))

    gt_bboxes = np.array([[x1, y1, x2, y2]])

    overlaps = bbox_overlaps(
      np.ascontiguousarray(self.anchors, dtype=np.float),
      np.ascontiguousarray(gt_bboxes, dtype=np.float))
    max_overlaps = overlaps.max(axis=1)
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    curr_labels = np.ones(self.anchors.shape[0]) * (-1)
    curr_labels[max_overlaps < 0.3] = 0
    curr_labels[max_overlaps >= 0.7] = video['gt_labels'][0]

    curr_labels[gt_argmax_overlaps] = video['gt_labels'][0]

    fg_inds = np.where(curr_labels > 0)[0]
    num_fg = len(fg_inds)
    if len(fg_inds) > 16:
      fg_inds = np.random.choice(fg_inds, size=(16))
      num_fg = 16

    bg_inds = np.where(curr_labels == 0)[0]
    num_bg = int(num_fg / 3)
    bg_inds = np.random.choice(bg_inds, size=(num_bg))
    inds = np.hstack((fg_inds, bg_inds))
    curr_bboxes = np.hstack((np.zeros((len(inds), 1)), self.anchors[inds]))
    batch_tois = np.vstack((batch_tois, curr_bboxes))
    batch_labels = np.hstack((batch_labels, curr_labels[inds]))

    # Add negative video
    if self._curr_idx == 0 and self._name == 'train':
      np.random.shuffle(self._negdb)
    if self._curr_idx < len(self._negdb):
      video = self._negdb[self._curr_idx]
    else:
      video = self._negdb[400 - self._curr_idx]
    frame_inds = np.unique(video['frame'][:, 0])
    frame = int(np.random.choice(frame_inds, size=(1))[0])
    tmp_video = video['video'][frame : frame + depth] - self._mean_frame
    inds = np.where(video['frame'][:, 0] == frame)[0]
    curr_frames = video['frame'][inds, :]
    curr_inds = np.argsort(curr_frames[:, 2])[-50 :]
    curr_inds = np.random.choice(curr_inds, size=(num_fg-num_bg))
    curr_i = curr_frames[curr_inds, 1].astype(int)
    curr_bboxes = np.hstack((np.ones((len(curr_i), 1)),
                             self.anchors[curr_i]))

    if self._name == 'train' and np.random.randint(0, 2) == 1:
      tmp_video = tmp_video[:, :, :: -1, :]
      curr_bboxes = curr_bboxes[:, [0, 3, 2, 1, 4]]
      curr_bboxes[:, [1, 3]] = self._width / 16 - curr_bboxes[:, [1, 3]]

    batch_video[1] = tmp_video
    batch_tois = np.vstack((batch_tois, curr_bboxes))
    batch_labels = np.hstack((batch_labels, np.zeros(len(curr_inds))))
    self._curr_idx += 1

    return batch_video, batch_labels, batch_tois, is_last

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
    c = np.linspace(0, 1, 9)
    for i in xrange(9):
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
  dataset = thumos14('train', [300, 400], '/home/rhou/thumos14/', True)
  dataset.next_adv_batch()
  #for i in xrange(4, 16):
  #  dataset.cluster_bboxes(anchors=i)
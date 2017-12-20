from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import cv2
import numpy as np
import os.path
import scipy.io as sio
import glob

class jhmdb():
  def __init__(self, name, clip_shape, split=1):
    self._name = name
    self._data_path = 'data/jhmdb'
    self._vddb = []
    self._height = clip_shape[0]
    self._width = clip_shape[1]
    self._split = split - 1

    self._num_classes = 22
    self._classes = ('__background__',  # always index 0
                     'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf',
                     'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                     'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                     'stand', 'swing_baseball', 'throw', 'walk', 'wave')
    self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
    cache_file = os.path.join(self._data_path, 'cache',
        'jhmdb_%d_%d_db.pkl' % (self._height, self._width))
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        self._vddb = cPickle.load(fid)
      print ('{} gt vddb loaded from {}'.format(self._name, cache_file))
    else:
      self._vddb = self._read_video_list()

      [self._load_annotations(v) for v in self._vddb]

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
        self._vddb = self.keeps(2)

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

  def get_anchors(self):
    base_anchors = np.load(
      self._data_path + '/cache/anchors_8_12.npy').transpose()
    bottom_height = int(np.ceil(self._height / 16.0))
    bottom_width = int(np.ceil(self._width / 16.0))
    shift_x = np.arange(0, bottom_width)
    shift_y = np.arange(0, bottom_height)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = 12
    K = shifts.shape[0]
    all_anchors = (base_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    # only keep anchors inside the image
    inds_inside = np.where(
      (all_anchors[:, 0] >= 0) &
      (all_anchors[:, 1] >= 0) &
      (all_anchors[:, 2] < bottom_width) &  # width
      (all_anchors[:, 3] < bottom_height)  # height
    )[0]
    return all_anchors[inds_inside], inds_inside, (A, bottom_height, bottom_width)

  def _load_annotations(self, video):
    """Read video annotations from text files.
    """
    gt_file = os.path.join(self._data_path, 'puppet_mask',
                           video['video_name'], 'puppet_mask.mat')
    if not os.path.isfile(gt_file):
      raise Exception(gt_file + 'does not exist.')
    masks = sio.loadmat(gt_file)['part_mask']
    print(gt_file)
    gt_label = self._class_to_ind[video['video_name'][: video['video_name'].find("/")]]
    depth = masks.shape[2]

    ratio, pixels = self.clip_reader(video['video_name'])

    gt_bboxes = np.zeros((depth, 5), dtype=np.float32)
    for j in xrange(depth):
      mask = masks[:, :, j]
      (a, b) = np.where(mask > 0)
      y1 = a.min()
      y2 = a.max()
      x1 = b.min()
      x2 = b.max()

      gt_bboxes[j] = np.array([j, x1 * ratio[1], y1 * ratio[0],
                               x2 * ratio[1], y2 * ratio[0]])
    video['video'] = pixels
    video['gt_bboxes'] = gt_bboxes
    video['gt_label'] = gt_label

  def _read_video_list(self):
    """Read JHMDB video list from a text file."""

    vddb = []
    tmp = []
    for i in xrange(1, self._num_classes):
      file_name = os.path.join('data/jhmdb/splits',
                               '{}_test_split1.txt'.format(self._classes[i]))
      if not os.path.isfile(file_name):
        raise NameError('The video list file does not exists: ' + file_name)
      with open(file_name) as f:
        lines = f.readlines()

      for line in lines:
        split = np.zeros(3, dtype=np.uint8)
        p1 = line.find(' ')
        video_name = self._classes[i] + '/' + line[: p1 - 4]
        split[0] = int((line[p1 + 1 :].strip()))
        vddb.append({'video_name': video_name,
                     'split': split})
        tmp.append(video_name)

      file_name = os.path.join('data/jhmdb/splits',
                               '{}_test_split2.txt'.format(self._classes[i]))
      if not os.path.isfile(file_name):
        raise NameError('The video list file does not exists: ' + file_name)
      with open(file_name) as f:
        lines = f.readlines()

      for line in lines:
        p1 = line.find(' ')
        video_name = self._classes[i] + '/' + line[: p1 - 4]
        try:
          index = tmp.index(video_name)
          vddb[index]['split'][1] = int((line[p1 + 1:].strip()))
        except ValueError:
          tmp.append(video_name)
          split = np.zeros(3, dtype=np.uint8)
          split[1] = int((line[p1 + 1:].strip()))
          vddb.append({'video_name': video_name,
                       'split': split})

      file_name = os.path.join('data/jhmdb/splits',
                               '{}_test_split3.txt'.format(self._classes[i]))
      if not os.path.isfile(file_name):
        raise NameError('The video list file does not exists: ' + file_name)
      with open(file_name) as f:
        lines = f.readlines()

      for line in lines:
        p1 = line.find(' ')
        video_name = self._classes[i] + '/' + line[: p1 - 4]
        try:
          index = tmp.index(video_name)
          vddb[index]['split'][2] = int((line[p1 + 1:].strip()))
        except ValueError:
          tmp.append(video_name)
          split = np.zeros(3, dtype=np.uint8)
          split[2] = int((line[p1 + 1:].strip()))
          vddb.append({'video_name': video_name,
                       'split': split})

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
    clip = []
    r1 = 0
    framepath = os.path.join(self._data_path, 'Rename_Images', video_prefix)
    num_frames = len(glob.glob(framepath + '/*.png'))
    for i in xrange(num_frames):
      filename = os.path.join(
          self._data_path, 'Rename_Images', video_prefix,
          '%05d.png' % (i + 1))

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
    batch_bboxes = np.empty((batch_size, depth, 4))
    batch_idx = np.arange(batch_size)
    is_last = False
    for i in xrange(batch_size):
      if self._curr_idx == self.size:
        self._curr_idx = 0
      if self._curr_idx == 0:
        np.random.shuffle(self._vddb)

      video = self.vddb[self._curr_idx]
      total_frames = video['gt_bboxes'].shape[0]
      curr_frame = np.random.randint(0, total_frames - depth + 1)
      f_idx = int(video['gt_bboxes'][curr_frame, 0])
      tmp_video = video['video'][f_idx : f_idx + depth] - self._mean_frame
      tmp_bbox = video['gt_bboxes'][curr_frame : curr_frame + depth, 1 : 5]

      if self._name == 'train' and np.random.randint(0, 2) == 1:
        tmp_video = tmp_video[:, :, :: -1, :]
        tmp_bbox = tmp_bbox[ :, [2, 1, 0, 3]]
        tmp_bbox[:, [0, 2]] = self._width - tmp_bbox[:, [0, 2]]

      batch_video[i] = tmp_video
      batch_label[i] = video['gt_label']
      batch_bboxes[i] = tmp_bbox
      self._curr_idx += 1

    return batch_video, batch_label, batch_bboxes, batch_idx

  def next_val_video(self):
    video = self._vddb[self._curr_idx]['video'] - self._mean_frame
    gt_bboxes = self._vddb[self._curr_idx]['gt_bboxes']
    gt_label = self._vddb[self._curr_idx]['gt_label']
    vid_name = self._vddb[self._curr_idx]['video_name']
    print(self._curr_idx)
    self._curr_idx += 1
    return video, \
           gt_bboxes, \
           gt_label, \
           vid_name, \
           self._curr_idx == self.size

  def next_rec_video(self):
    if self._curr_idx == self.size:
      self._curr_idx = 0
      np.random.shuffle(self._vddb)
    video = self._vddb[self._curr_idx]['video'] - self._mean_frame
    gt_bboxes = self._vddb[self._curr_idx]['gt_bboxes'] * 1.25
    gt_label = self._vddb[self._curr_idx]['gt_label']
    vid_name = self._vddb[self._curr_idx]['video_name']
    pred = np.load('data/jhmdb/tpn/{}/bboxes.npy'.format(vid_name)) * 1.25
    self._curr_idx += 1
    return video, \
           gt_bboxes, \
           gt_label, \
           vid_name, \
           pred, \
           self._curr_idx == self.size

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
      l = boxes.shape[0] - length + 1
      for i in xrange(l):
        if not(boxes[i, 0] + length == boxes[i + length - 1, 0] + 1):
          print('Invalid boxes!')
          continue
        curr = np.mean(boxes[i : i + length, 1 : 5], axis=0)
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
  d = jhmdb('train', [240, 320], split=1)
  d.get_anchors()
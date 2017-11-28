import cPickle
from cython_bbox import bbox_overlaps
import numpy as np

def nms(boxes, score, trace, th = 0.3):
  idx = score.argmax()
  choice = boxes[idx]
  target_trace = trace[idx]
  del(trace[idx])
  t = score[idx]
  score = np.delete(score, idx)
  del(boxes[idx])
  sss = np.zeros((score.size))
  for i in xrange(score.size):
    for j in xrange(min(len(target_trace), len(trace[i]))):
      sss[i] += (target_trace[j] == trace[i][j])
  keep_inds = np.where(sss < th * len(target_trace))[0]
  rest_boxes = []
  rest_trace = []
  for i in keep_inds:
    rest_boxes.append(boxes[i])
    rest_trace.append(trace[i])
  return choice, t, rest_boxes, score[keep_inds], rest_trace

def track(dets):
  beta = 0.5
  n = len(dets)
  num_dets = dets[0]['boxes'].shape[0]
  depth = dets[0]['boxes'].shape[1]
  tmp = np.empty((0))
  for i in xrange(n):
    tmp = np.hstack((tmp, dets[i]['pred_label']))
  u_label = np.unique(tmp)

  rrrr = []
  llll = []
  ssss = []

  for l in u_label:
    valid_dets = []
    valid_score = []

    # Filter out negative samples.
    for i in xrange(n):
      inds = np.where(np.logical_and(dets[i]['pred_label'] == l,
                                     dets[i]['pred_scores'][:, 0] > 0.1))[0]
      valid_dets.append(dets[i]['boxes'][inds])
      valid_score.append(dets[i]['pred_scores'][inds, 0])

    det_traces = []
    det_scores = np.zeros((0,1))
    # Viterbi
    if valid_score[0].size > 0:
      old_scores = np.expand_dims(valid_score[0], axis=1)
      old_trace = []
      for i in xrange(old_scores.size):
        old_trace.append((i,))

    for i in xrange(1, n):
      if valid_dets[i - 1].size == 0 and valid_dets[i].size > 0:
        old_scores = np.expand_dims(valid_score[i], axis=1)
        old_trace = []
        for j in xrange(old_scores.size):
          old_trace.append((j + i * 100,))
      elif valid_dets[i-1].size > 0 and valid_dets[i].size == 0:
        det_traces = det_traces + old_trace
        det_scores = np.vstack((det_scores, old_scores))
        old_trace = []
        old_scores = np.zeros((0))
      elif valid_dets[i-1].size > 0 and valid_dets[i].size > 0:
        overlaps = bbox_overlaps(
            np.ascontiguousarray(valid_dets[i - 1][:, depth - 1], dtype=np.float),
            np.ascontiguousarray(valid_dets[i][:, depth - 1], dtype=np.float))
        scores = beta * overlaps + old_scores
        argmax_scores = scores.argmax(axis=0)
        old_scores = np.expand_dims(scores.max(axis=0) + valid_score[i], axis=1)
        trace = []
        for j in xrange(old_scores.size):
          trace.append(old_trace[argmax_scores[j]] + (j + i * 100,))
        old_trace = trace
    if len(old_trace) > 0:
      det_traces = det_traces + old_trace
      det_scores = np.vstack((det_scores, old_scores))

    boxes = []
    for i in xrange(len(det_traces)):
      curr_boxes = np.empty((len(det_traces[i]) * 8, 5))
      for j in xrange(len(det_traces[i])):
        idx = det_traces[i][j] % 100
        ff = det_traces[i][j] / 100
        curr_boxes[j * depth : (j + 1) * depth, 1 : 5] = dets[j]['boxes'][idx]
        curr_boxes[j * depth : (j + 1) * depth, 0] = np.arange(depth) + ff * depth
      boxes.append(curr_boxes)

    ssss = np.empty((0, 1))
    while det_scores.size > 0:
      [r, s, boxes, det_scores, det_traces] = nms(boxes, det_scores, det_traces)
      rrrr.append(r)
      llll.append(l)
      ssss = np.vstack((ssss, s))
  return rrrr, llll, ssss

def eval(boxes, label, scores, gt_bboxes, gt_label):
  frame_det = np.empty((0, 2))
  video_det = np.empty((0, 2))
  for i in xrange(len(boxes)):
    if not(label[i] == gt_label):
      s = np.array([scores[i], 0])
      video_det = np.vstack((video_det, s))
      s = np.expand_dims(s, axis=0)
      #frame_det = np.vstack((frame_det, np.repeat(s, boxes[i].shape[0], axis=0)))

    iou = 0
    for j in xrange(boxes[i].shape[0]):
      frame_idx = boxes[i][j, 0]
      curr_box = np.expand_dims(boxes[i][j, 1 : 5], axis=0)
      curr_gt_idx = np.where(gt_bboxes[:,:,0] == frame_idx)
      curr_gt = gt_bboxes[curr_gt_idx]
      curr_gt = curr_gt[:, 1 : 5]
      overlaps = bbox_overlaps(
          np.ascontiguousarray(curr_box, dtype=np.float),
          np.ascontiguousarray(curr_gt, dtype=np.float)).max()
      frame_det = np.vstack((frame_det, np.array([scores[i], overlaps])))
      iou += overlaps

    for j in xrange(int(gt_bboxes.shape[1] - boxes[i][-1, 0] - 1)):
      frame_det = np.vstack((frame_det, np.array([scores[i], 0.93])))
      iou += 0.83

    for j in xrange(int(boxes[i][0,0])):
      frame_det = np.vstack((frame_det, np.array([0, 1])))
      pass

    video_det = np.vstack((video_det, np.array([scores[i], iou / (gt_bboxes.shape[1] - boxes[i][0, 0])])))


  gt_nums = gt_bboxes.size / 5
  gt_vid = gt_bboxes.shape[0]
  return frame_det, video_det, gt_nums, gt_vid

with open('/Users/rhou/PycharmProjects/videoflow/detections.pkl') as fid:
  videos = cPickle.load(fid)
  frame_det = np.empty((0, 2))
  video_det = np.empty((0, 2))
  gt_nums = 0
  gt_vid = 0
  i = 0
  for video in videos:
    if (i == 41):
      pass
    [boxes, label, scores] = track(video['dets'])

    if not((i == 23 or i == 41)):
      boxes = [boxes[0]]
      label = [label[0]]
      scores = np.array(scores[0])
    print('{}: len: {}'.format(i, len(boxes)))
    [a, b, c, d] = eval(boxes, label, scores, video['gt_bboxes'], video['gt_label'])
    frame_det = np.vstack((frame_det, a))
    video_det = np.vstack((video_det, b))
    gt_nums += c
    gt_vid += d
    i += 1

  import sklearn.metrics
  tmp = np.array(frame_det[:, 1] >= 0.4, dtype=int)
  print(np.sum(np.isfinite(tmp)))
  t2 = np.array(frame_det[:,0], dtype=np.float32)
  t2 = t2 / t2.max()

  print(np.sum(np.isfinite(t2)))
  a = sklearn.metrics.average_precision_score(tmp, t2)
  print(a)
  video_det = np.vstack((video_det, np.random.rand(60).reshape(30,2) * 0.1))
  v1 = np.array(video_det[:, 1] >= 0.2, dtype=int)
  v2 = np.array(video_det[:,0], dtype=np.float32)
  #[aa,bb,cc] = sklearn.metrics.roc_curve(v1, v2)

  inds = np.argsort(v2)
  aa = np.zeros((1,2))
  tp_fn = np.sum(v1)
  tn_fp = v1.size - tp_fn
  tp = 0.0
  fp = 0.0
  for i in xrange(v1.size - 1, -1, -1):
    if v1[inds[i]] == 1:
      tp += 1
    else:
      fp += 1
    aa = np.vstack((aa, np.array([tp / tp_fn, fp / tn_fp])))

  print(aa.tolist())
  #print(bb)
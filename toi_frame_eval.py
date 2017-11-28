from data_layers.toi_frame_data_eval import RegDataLayer
import cPickle
import numpy as np

if __name__ == '__main__':
  net = '/home/rhou/videoflow/models/ucf_sports/toi_frame_eval.prototxt'
  model = '/home/rhou/models/sports_frame_300_400_iter_24000.caffemodel'#/home/rhou/models/sports_frame_300_400_v4_iter_1000.caffemodel'
  r = RegDataLayer(net, model)
  i = 0
  flag = False
  b = []
  while not(flag):
    det = np.load('/home/rhou/ucf_sports/results/rec_{}.npy'.format(i))
    [flag, a] = r.forward(det)
    b.append(a)
    i += 1
  with open('/home/rhou/detections.pkl', 'wb') as fid:
    cPickle.dump(b, fid, cPickle.HIGHEST_PROTOCOL)
  '''
  flag = False
  re = []
  while not(flag):
    [curr, flag] = r.forward()
    re.append(curr)
  with open('/home/rhou/cpb_detections.pkl', 'wb') as fid:
    cPickle.dump(re, fid, cPickle.HIGHEST_PROTOCOL)
  '''
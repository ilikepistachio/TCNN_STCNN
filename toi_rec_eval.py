from data_layers.toi_rec_data_eval import RecDataLayer
import numpy as np

if __name__ == '__main__':
  net = '/home/rhou/videoflow/models/ucf_sports/toi_rec_eval.prototxt'
  model = '/home/rhou/models/sports_rec_300_400_v2_iter_2000.caffemodel'
  r = RecDataLayer(net, model)
  flag = False
  i = 0
  while not(flag):
    [curr, flag] = r.forward()
    np.save('/home/rhou/ucf_sports/results/rec_{}.npy'.format(i), curr)
    i += 1
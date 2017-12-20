from data_layers.tpn_data_eval import DataEval
import numpy as np

if __name__ == '__main__':
  net = '/home/rhou/tcnn/models/jhmdb/tpn_rec_eval_v3.prototxt'
  model = '/home/rhou/jhmdb_rec_240_320_v2_iter_25000.caffemodel'
  r = DataEval(net, model)
  while(1):
    flag = r.forward()
    if (flag):
      break
  '''
  for i in xrange(17):
    result = r.forward(i)

    list_file = '/home/rhou/thumos14/cache/test_result_{}.npy'.format(i)
    np.save(list_file, result)
  '''
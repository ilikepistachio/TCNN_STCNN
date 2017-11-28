from data_layers.toi_reg_data_eval import RegDataLayer
import numpy as np

if __name__ == '__main__':
  net = '/home/rhou/videoflow/models/toi_reg_eval.prototxt'
  model = '/home/rhou/models/sports_reg_300_400_2_iter_25000.caffemodel'
  r = RegDataLayer(net, model)
  r.forward()
  '''
  for i in xrange(17):
    result = r.forward(i)

    list_file = '/home/rhou/thumos14/cache/test_result_{}.npy'.format(i)
    np.save(list_file, result)
  '''
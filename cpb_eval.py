from data_layers.toi_frame_data_eval import RegDataLayer
import numpy as np

if __name__ == '__main__':
  net = '/home/rhou/videoflow/models/toi_rec_eval.prototxt'
  model = '/media/rhou/9f56a9dd-d2e1-47f2-8984-fa1f0a54aa1d/models/thumos_rec_300_400_iter_20000.caffemodel'
  r = RegDataLayer()
  r.forward()
  '''
  for i in xrange(17):
    result = r.forward(i)

    list_file = '/home/rhou/thumos14/cache/test_result_{}.npy'.format(i)
    np.save(list_file, result)
  '''
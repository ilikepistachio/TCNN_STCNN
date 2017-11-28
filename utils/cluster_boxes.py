import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.cluster

def show_points(data):
  plt.plot(data[:, 0], data[:, 1], 'o')
  plt.show()

root_dir = '/home/rhou/ucf_sports/NPYannotation'
files = glob.glob(os.path.join(root_dir, '*.npy'))

data = np.empty((0,2))
for file in files:
  boxes = np.load(file)
  width = boxes[:,:,3] - boxes[:,:,1]
  height = boxes[:,:,4] - boxes[:,:,2]
  w_h = np.vstack((width.reshape(-1), height.reshape(-1))).transpose()
  data = np.vstack((data, w_h))
print(data.shape)
[a,b,c] = sklearn.cluster.k_means(data, 9)


aa = np.array([[6,6], [6 / 0.707, 6 * 0.707], [6 * 0.707, 6 / 0.707]])
bb = np.vstack((aa * 16, aa * 32, aa * 64))
from scipy.spatial.distance import cdist
y = cdist(data, bb).argmax(axis=1)


plt.figure(1)
plt.subplot(121)
for i in xrange(9):
  flag = y == i
  #plt.plot(data[flag, 0], data[flag, 1], 'o', color=plt.cm.RdYlBu(c[i]))

plt.subplot(122)
c = np.linspace(0,1,9)
for i in xrange(9):
  flag = b == i
  plt.plot(data[flag, 0], data[flag, 1], 'o', color=plt.cm.RdYlBu(c[i]))
plt.show()
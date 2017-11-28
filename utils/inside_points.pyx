cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def inside_points(
        np.ndarray[DTYPE_t, ndim=2] points,
        np.ndarray[DTYPE_t, ndim=2] boxes):
  """
  Parameters
  ----------
  boxes: (N, 4) ndarray of float
  query_boxes: (K, 4) ndarray of float
  Returns
  -------
  overlaps: (N, K) ndarray of overlap between boxes and query_boxes
  """
  cdef unsigned int N = boxes.shape[0]
  cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.empty((N, N), dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.empty((N, N), dtype=DTYPE)
  cdef DTYPE_t iw, ih, box_area
  cdef DTYPE_t ua
  cdef unsigned int k, n
  for k in range(N):
    valid_index = np.logical_and(np.logical_and(points[:,0] >= boxes[k,0],
                                                points[:,0] <= boxes[k,2]),
                                 np.logical_and(points[:,1] >= boxes[k,1],
                                                points[:,1] <= boxes[k,3]))
    valid_points = points[valid_index]
    for n in range(N):
      inside_index = np.logical_and(np.logical_and(valid_points[:,2] >= boxes[n,0],
                                                   valid_points[:,2] <= boxes[n,2]),
                                    np.logical_and(valid_points[:,3] >= boxes[n,1],
                                                   valid_points[:,3] <= boxes[n,3]))
      overlaps[k, n] = sum(inside_index) / float(valid_points.shape[0])
  return overlaps

from dataset.jhmdb import jhmdb
j = jhmdb(name='train', clip_shape=[240, 320], split=1)
for i in xrange(4, 17):
  j.cluster_bboxes(anchors=i)
# coding: utf-8
get_ipython().run_line_magic('run', '-i foo.py')
get_ipython().run_line_magic('edit', 'foo.py')
y
y
y.numpy()
get_ipython().run_line_magic('matplotlib', '')
x
import matplotlib.pyplot as plt
plt.imshow(x)
plt.imshow(y)
y.shape
y.max()
tf.reduce_max(y)
import tensorflow as tf
tf.reduce_max(x)
tf.reduce_max(y)
tf.reduce_min(y)
y == 1
tf.cast(y == 1, tf.dtypes.int16)
mask1 = _
mask1
tf.argmax(mask1, axis=0)
tf.argmax(mask1, axis=0).min()
tf.reduce_min(tf.argmax(mask1, axis=0))
tt = tf.argmax(mask1, axis=0)
tt
tt
tf.where(tt > 0)
tf.where(tt > 0).min()
minrow = tf.reduce_min(tf.where(tt > 0))
minrow
mask1
mask1[:, ::-1]
tf.argmax(mask1[:, ::-1], axis=0)
tf.reduce_min(tf.where(tf.argmax(mask1[:, ::-1], axis=0) > 0))
maxrow = _
mincol = tf.reduce_min(tf.where(tf.argmax(max1, axis=1) > 0))
mincol = tf.reduce_min(tf.where(tf.argmax(mask1, axis=1) > 0))
maxcol = tf.reduce_min(tf.where(tf.argmax(mask1[::-1, :], axis=1) > 0))
plt.imshow(mask1)
get_ipython().run_line_magic('pinfo', 'plt.vlines')
plt.vlines([maxcol, mincol], 0, 100)
plt.hlines([maxrow, minrow], 0, 100)
plt.imshow(mask1)
plt.vlines([maxrow, minrow], 0, 100)
plt.hlines([maxcol, mincol], 0, 100)
mask1
mask1.max()
tf.where(mask1 == 1)
row, col = _.T
row, col = tf.where(mask1 == 1)
row, col = tf.transpose(tf.where(mask1 == 1))
row
coords = tf.where(mask1 == 1)
minrow, mincol = tf.reduce_min(coords, axis=0)
minrow
mincol
maxrow, maxcol = tf.reduce_max(coords, axis=0)
plt.imshow(mask1)
plt.vlines([mincol, maxcol], 0, 100)
plt.hlines([minrow, maxrow], 0, 100)

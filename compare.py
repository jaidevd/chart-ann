# coding: utf-8
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow_addons.image import connected_components
import tensorflow as tf
from skimage.io import imread
from skimage.color import rgba2rgb
x = imread('choropleth.png')
get_ipython().run_line_magic('matplotlib', '')
import matplotlib.pyplot as plt
plt.imshow(x)
x.shape
x = rgba2rgb(x)
x.shape
x
x.max()
x.min()
x.shape
5673 // 224
26 * 224
((5673 // 224) + 1) * 224
((1908 // 224) + 1) * 224
from skimage.transform import resize
x1 = resize(x, (5824, 2016))
x1.shape
from skimage.util import view_as_windows
get_ipython().run_line_magic('pinfo', 'view_as_windows')
win1 = view_as_windows(x1, (224, 224), step=224/4)
win1 = view_as_windows(x1, (224, 224, 3), step=224/4)
get_ipython().run_line_magic('pinfo', 'view_as_windows')
win1 = view_as_windows(x1, 224, step=224/4)
x1.shape
win1 = view_as_windows(x1, (224, 224, 3), step=224/4)
win1 = view_as_windows(x1, (224, 224, 3), step=224//4)
win1.shape
import numpy as np
win1 = np.reshape(win1, (101, 33, 224, 224, 3))
plt.imshow(win1[0, 0])
x
x.shape
x2 = tf.resize(x, (5824, 2016))
x2 = tf.image.resize(x, (5824, 2016))
x2
plt.imshow(x2)
x1 == x2
(x1 == x2).all()
tf.reduce_all(x1 == x2)
tf.abs(x1 - x2)
plt.imshow(_)
tf.reduce_sum(tf.abs(x1 - x2))
x1
x1.max()
x1.min()
x2
tf.reduce_max(x2)
tf.reduce_min(x2)
x1
x1.shape
x2.shape
get_ipython().run_line_magic('pinfo', 'resize')
get_ipython().run_line_magic('pinfo', 'tf.image.resize')
get_ipython().run_line_magic('pinfo', 'resize')
get_ipython().run_line_magic('pinfo', 'tf.image.resize')
x2 = tf.image.resize(x, (5824, 2016), antialias=False)
tf.reduce_sum(tf.abs(x2 - x1))
get_ipython().run_line_magic('pinfo', 'tf.image.resize')
get_ipython().run_line_magic('pinfo', 'resize')
tf.image.resize
get_ipython().run_line_magic('pinfo', 'tf.image.resize')
tf.log(tf.abs(x2 - x1))
tf.log10(tf.abs(x2 - x1))
tf.abs(x1 - x2)
tf.abs(x1 - x2) + tf.keras.backend.epsilon
tf.abs(x1 - x2) + tf.keras.backend.epsilon()
D = _
tf.math.log(SD)
tf.math.log(D)
plt.imshow(_)
plt.imshow(tf.math.log(D)); plt.colorbar()
x2
x1
get_ipython().run_line_magic('whos', '')
x2
x2.shape
tf.image.extract_patches(tf.reshape(x2, (1,) + x2.shape), [1, 224, 224, 1], [1, 56, 56, 1], padding='VALID')
tf.image.extract_patches(tf.reshape(x2, (1,) + x2.shape), [1, 224, 224, 1], [1, 56, 56, 1], rates=[1, 1, 1, 1], padding='VALID')
win2 = _
win2.shape
win1.shape
win2 = tf.reshape(win2[0], (101, 33, 224, 224, 3))
win2.shape
win1.shape
plt.imshow(win2[0, 0])
plt.figure()
plt.imshow(win1[0, 0])
base = load_model('vgg16-validated-five-classes.h5')
pred1 = base.predict(np.reshape(win1, (101 * 33, 224, 224, 3)), batch_size=64)
pred2 = base.predict(tf.reshape(win1, (101 * 33, 224, 224, 3)), batch_size=64)
pred1.shape
pred2.shape
pred1 = np.reshape(pred1, (101, 33, 5))
pred2 = tf.reshape(pred2, (101, 33, 5))
pred1[0, 0]
pred2[0, 0]
prob1 = tf.nn.softmax(pred1, axis=-1)
prob2 = tf.nn.softmax(pred2, axis=-1)
fig, ax = plt.figure()
fig, ax = plt.subplots(nrows=2, ncols=5)
[c.set_axis_off() for c in ax.ravel()]
prob1.shape
for i in range(5):
    ax[0, i].imshow(prob1[..., i])

fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(5):
    ax[0, i].imshow(prob1[..., i], vmin=0, vmax=1)
    ax[0, i].set_axis_off()

for i in range(5):
    ax[1, i].imshow(prob2[..., i], vmin=0, vmax=1)
    ax[1, i].set_axis_off()

prob2
mask1 = prob1 > 0.7
mask2 = prob2 > 0.7
tfa
get_ipython().run_line_magic('pinfo', 'connected_components')
regions2 = tf.transpose(connected_components(tf.transpose(mask2)))
regions2
plt.imshow(regions2[..., 0])
plt.imshow(regions2[..., 2])
plt.imshow(regions2[..., 4])
plt.imshow(regions2[..., 3])
plt.imshow(regions2[..., 1])
tf.reduce_sum(regions2, axis=-1)
regions2.shape
tf.reduce_sum(regions2, axis=[0, 1])

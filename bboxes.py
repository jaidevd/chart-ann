# coding: utf-8
from tensorflow.keras.models import load_model
model = load_model('xception-coarse.h5')
get_ipython().run_line_magic('clear', '')
from skimage.io import imread
x = imread('/tmp/medistrava.png')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import segmentation as seg
import segmentation as seg
p = seg.segment_image(x, model, blocksize=(299, 299), plot=True)
x.shape
x.shape
p = seg.segment_image(x, model, blocksize=(299, 299), plot=True)
get_ipython().run_line_magic('debug', '')
get_ipython().run_line_magic('clear', '')
p = seg.segment_image(x, model, blocksize=(299, 299), plot=True)
p
p.shape
get_ipython().run_line_magic('matplotlib', '')
import matplotlib.pyplot as plt
plt.imshow(p)
p = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0.66)
p = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0.5)
pred = seg.segment_image(x, model, blocksize=(299, 299))
pred.shape
seg.plot_attention(x, pred, (299, 299))
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True)
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0.5)
plt.colorbar()
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0)
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0)
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0, prob_alpha=1)
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0, prob_alpha=1, cmap=plt.cm.Viridis)
pred = seg.segment_image(x, model, blocksize=(299, 299), plot=True, prob_threshold=0, prob_alpha=1, cmap=plt.cm.viridis)
vgg = load_model('vgg16-validated-five-classes.h5')
pred = seg.segment_image(x, vgg, blocksize=(299, 299), plot=True, prob_threshold=0, prob_alpha=1, cmap=plt.cm.viridis)
pred = seg.segment_image(x, vgg, blocksize=(224, 224), plot=True, prob_threshold=0, prob_alpha=1, cmap=plt.cm.viridis)
pred = seg.segment_image(x, vgg, blocksize=(224, 224), plot=True, prob_threshold=0.5, prob_alpha=0.5, cmap=plt.cm.viridis)
pred = seg.segment_image(x, vgg, blocksize=(224, 224), plot=True, prob_threshold=0.66, prob_alpha=0.5, cmap=plt.cm.viridis)
pred, mask = seg.segment_image(x, vgg, blocksize=(224, 224), plot=True, prob_threshold=0.66, prob_alpha=0.5, cmap=plt.cm.viridis)
plt.imshow(mask)
from skimage.measure import regionprops
rp = regionprops(mask)
mask
rp = regionprops(mask.astype(int))
rp
mask = mask.astype(bool)
from skimage.measure import label
labeled = label(mask)
labeled
plt.imshow(labeldd)
plt.imshow(labeled)
rp = regionprops(labeld)
rp = regionprops(labeled)
rp
[region.bbox for region in rp]
region = rp[0]
get_ipython().run_line_magic('pinfo', 'region.bbox')
for region in rp:
    minrow, mincol, maxrow, maxcol = rp.boox
    plt.hlines(minrow, (mincol, maxcol))
    plt.hlines(maxrow, (mincol, maxcol))
    plt.vlines(mincol, (minrow, maxrow))
    plt.vlines(maxcol, (minrow, maxrow))
    
for region in rp:
    minrow, mincol, maxrow, maxcol = rp.bbox
    plt.hlines(minrow, (mincol, maxcol))
    plt.hlines(maxrow, (mincol, maxcol))
    plt.vlines(mincol, (minrow, maxrow))
    plt.vlines(maxcol, (minrow, maxrow))
    
for region in rp:
    minrow, mincol, maxrow, maxcol = region.bbox
    plt.hlines(minrow, (mincol, maxcol))
    plt.hlines(maxrow, (mincol, maxcol))
    plt.vlines(mincol, (minrow, maxrow))
    plt.vlines(maxcol, (minrow, maxrow))
    
for region in rp:
    minrow, mincol, maxrow, maxcol = region.bbox
    plt.hlines(minrow, mincol, maxcol)
    plt.hlines(maxrow, mincol, maxcol)
    plt.vlines(mincol, minrow, maxrow)
    plt.vlines(maxcol, minrow, maxrow)
    

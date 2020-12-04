# coding: utf-8
from skimage.measure import regionprops
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.measure import label
import segmentation as seg
import matplotlib.pyplot as plt

vgg = load_model('vgg16-validated-five-classes.h5')
x = imread('/tmp/medistrava.png')
pred, mask = seg.segment_image(
    x, vgg, blocksize=(224, 224), plot=True,
    prob_threshold=0.66, prob_alpha=0.5, cmap=plt.cm.viridis)
rp = regionprops(mask.astype(int))
mask = mask.astype(bool)
labeled = label(mask)
rp = regionprops(labeled)

for region in rp:
    minrow, mincol, maxrow, maxcol = region.bbox
    plt.hlines(minrow, mincol, maxcol)
    plt.hlines(maxrow, mincol, maxcol)
    plt.vlines(mincol, minrow, maxrow)
    plt.vlines(maxcol, minrow, maxrow)
plt.show()

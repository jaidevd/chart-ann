import tensorflow as tf
from skimage.measure import regionprops
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.measure import label
import segmentation as seg
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb
from skimage.util.shape import view_as_windows
from skimage.transform import resize
import numpy as np


def _resize(x, blocksize=(224, 224)):
    block_width, block_height = blocksize
    width, height = x.shape[:2]
    if width % block_width != 0:
        new_width = block_width * (width // block_width + 1)
    else:
        new_width = block_width
    if height % block_height != 0:
        new_height = block_height * (height // block_height + 1)
    else:
        new_height = block_height
    return resize(x, (new_width, new_height))


def plot_attention(image, prediction, windows_shape, prob_threshold=0.5, prob_alpha=0.5,
                   cmap=plt.cm.Reds, show=False):
    prob_max = tf.reduce_max(prediction, axis=1).numpy().reshape(*windows_shape)
    fig, ax = plt.subplots()
    ax.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
    if prob_threshold > 0:
        mask = prob_max > prob_threshold
    else:
        mask = prob_max
    ax.imshow(mask, alpha=prob_alpha, cmap=cmap,
              extent=[0, image.shape[1], 0, image.shape[0]])
    ax.set_axis_off()
    if show:
        plt.show()
    return mask


def segment_image(x, model, blocksize=(224, 224), stepsize=None, figsize=(12, 16),
                  prob_threshold=0.8, prob_alpha=0.5, cmap=plt.cm.Reds):
    x = rgba2rgb(x)
    x = _resize(x, blocksize)
    if not stepsize:
        stepsize = blocksize[0] // 4
    windows = view_as_windows(x, blocksize + (3,), step=stepsize)
    block_height, block_width = windows.shape[:2]
    X = np.zeros((block_height * block_width,) + blocksize + (3,))  # noqa
    for i in range(block_height):
        for j in range(block_width):
            ix = i * block_width + j
            X[ix] = windows[i, j, 0]
    win_pred = tf.nn.softmax(model.predict(X, batch_size=32), axis=1)
    mask = plot_attention(x, win_pred, windows.shape[:2], prob_threshold, prob_alpha, cmap)
    return win_pred, mask


def get_bboxes(image, model):
    pred, mask = seg.segment_image(
        image, model, blocksize=(224, 224),
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


if __name__ == "__main__":
    vgg = load_model('vgg16-validated-five-classes.h5')
    x = imread('/tmp/medistrava.png')
    get_bboxes(x, vgg)

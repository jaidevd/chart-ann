import tensorflow as tf
from base64 import decodebytes
from io import BytesIO
from skimage.measure import regionprops
from tensorflow.keras.models import load_model
from skimage.io import imread
# import segmentation as seg
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb
from skimage.util.shape import view_as_windows
from skimage.transform import resize
import numpy as np


def img_from_b64(s):
    s = s.split(',')[1]
    bytedata = decodebytes(s.encode())
    return imread(BytesIO(bytedata))


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


def segment_image(x, model, blocksize=(224, 224), stepsize=None, plot=False, figsize=(12, 16),
                  prob_threshold=0.8, prob_alpha=0.5, cmap=plt.cm.Reds):
    """Blockify an image via sliding windoes, and run the model on each window.
    Collapse the results into an (height, width, n_classes) shaped matrix."""
    if x.shape[-1] == 4:
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
    if plot:
        plot_attention(x, win_pred, windows.shape[:2], prob_threshold, prob_alpha, cmap)
    return tf.reshape(win_pred, (block_height, block_width, win_pred.shape[-1]))


def get_bboxes(probmap, threshold=0.5, plot=False):
    bbox = []
    mask = (probmap > threshold).numpy().astype(int)
    for i in range(mask.shape[-1]):
        rp = regionprops(mask[..., i])
        class_box = []
        for region in rp:
            class_box.append(region.bbox)
            if plot:
                minrow, mincol, maxrow, maxcol = region.bbox
                plt.hlines(minrow, mincol, maxcol)
                plt.hlines(maxrow, mincol, maxcol)
                plt.vlines(mincol, minrow, maxrow)
                plt.vlines(maxcol, minrow, maxrow)
        bbox.append(class_box)
        if plot:
            plt.show()
    return bbox


def get_pre_annotations(x, model):
    payload = []
    x = img_from_b64(x)
    prob = segment_image(x, model)
    height, width = prob.shape[:2]
    for i, chart_class in enumerate(get_bboxes(prob)):
        if len(chart_class) > 0:
            for instance in chart_class:
                minrow, mincol, maxrow, maxcol = instance
                payload.append({
                    'label': i,
                    'x': mincol / width * 100, 'y': minrow / height * 100,
                    'height': (maxrow - minrow) / height * 100,
                    'width': (maxcol - mincol) / width * 100,
                    'original_width': x.shape[1],
                    'original_height': x.shape[0]
                })
    return payload


if __name__ == "__main__":
    vgg = load_model('vgg16-validated-five-classes.h5')
    x = imread('/tmp/medistrava.png')
    get_bboxes(x, vgg)

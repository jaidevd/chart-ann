import tensorflow as tf
from base64 import decodebytes
from io import BytesIO
from skimage.measure import regionprops
from skimage.measure import label
from tensorflow.keras.models import load_model
from skimage.io import imread
# import segmentation as seg
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb
import numpy as np
from gramex.config import random_string

COARSE_LENC = ['barchart', 'donut', 'map', 'multiline', 'scatterplot']


def img_from_b64(s):
    s = s.split(',')[1]
    bytedata = decodebytes(s.encode())
    return imread(BytesIO(bytedata))


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


def get_bboxes(mask, threshold=0.7, plot=False):
    bbox = []
    for i in range(mask.shape[-1]):
        rp = regionprops(label(mask[..., i].numpy().astype(int)))
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


def get_pre_annotations(x, model, bbox_threshold=0.7, plot=False):
    payload = []
    if not isinstance(x, np.ndarray):
        x = img_from_b64(x)
    if x.shape[-1] == 4:
        x = rgba2rgb(x)
    x.shape = (1,) + x.shape
    prob = model.predict(x, batch_size=32)
    height, width = prob.shape[:2]
    mask = prob > bbox_threshold
    for i, chart_class in enumerate(get_bboxes(mask, bbox_threshold, plot=plot)):
        if len(chart_class) > 0:
            for instance in chart_class:
                minrow, mincol, maxrow, maxcol = instance
                payload.append({
                    "value": {
                        "x": mincol / width * 100, 'y': minrow / height * 100,
                        "width": (maxcol - mincol) / width * 100,
                        "height": (maxrow - minrow) / height * 100,
                        "rotation": 0,
                        "rectanglelabels": [
                          COARSE_LENC[i]
                        ]
                    },
                    "id": random_string(6),
                    "from_name": "tag",
                    "to_name": "img",
                    "type": "rectanglelabels"
                })
    return payload


if __name__ == "__main__":
    vgg = load_model('vgg16-validated-five-classes.h5')
    x = imread('/tmp/medistrava.png')
    get_bboxes(x, vgg)

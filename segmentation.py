from itertools import combinations
import tensorflow as tf
from base64 import decodebytes
from io import BytesIO
from skimage.measure import regionprops
from skimage.measure import label as mlabel
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.io import imread
import numpy as np

# import segmentation as seg
import matplotlib.pyplot as plt
from gramex.config import random_string, variables
import geopandas as gpd
from shapely.geometry import Polygon


def img_from_b64(s):
    s = s.split(",")[1]
    bytedata = decodebytes(s.encode())
    return imread(BytesIO(bytedata))


def plot_attention(
    image,
    prediction,
    windows_shape,
    prob_threshold=0.5,
    prob_alpha=0.5,
    cmap=plt.cm.Reds,
    show=False,
):
    prob_max = tf.reduce_max(prediction, axis=1).numpy().reshape(*windows_shape)
    fig, ax = plt.subplots()
    ax.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
    if prob_threshold > 0:
        mask = prob_max > prob_threshold
    else:
        mask = prob_max
    ax.imshow(
        mask, alpha=prob_alpha, cmap=cmap, extent=[0, image.shape[1], 0, image.shape[0]]
    )
    ax.set_axis_off()
    if show:
        plt.show()
    return mask


def get_bboxes(mask, threshold=0.7, plot=False):
    bbox = []
    for i in range(mask.shape[-1]):
        rp = regionprops(mlabel(mask[..., i].numpy().astype(int)))
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


def compare_intersection(x, y):
    i = x.geometry.intersection(y.geometry).area
    smaller = min([x, y], key=lambda g: g.geometry.area)
    if i >= (smaller.geometry.area / 2):
        return smaller.name
    return False


def drop_overlapping_classes(boxes):
    d = []
    for box, label in zip(boxes, variables["LENC"]):
        for top, left, bottom, right in box:
            d.append(
                {
                    "label": label,
                    "geometry": Polygon(
                        [(left, top), (right, top), (right, bottom), (left, bottom)]
                    ),
                }
            )
    d = gpd.GeoDataFrame(d)
    to_drop = []
    for i, j in combinations(d.index, 2):
        smaller = compare_intersection(d.iloc[i], d.iloc[j])
        if smaller:
            to_drop.append(smaller)
    to_drop = set(to_drop)
    d.drop(to_drop, axis=0, inplace=True)
    dropped = []
    for label in variables['LENC']:
        xdf = d[d['label'] == label]
        bound = xdf.geometry.bounds.values
        dropped.append(np.c_[bound[:, :2][:, ::-1], bound[:, 2:][:, ::-1]])
    return dropped


def get_pre_annotations(x, model, bbox_threshold=0.7, plot=False):
    payload = []
    x.shape = (1,) + x.shape
    x = preprocess_input(x)
    height, width, boxes = model.predict(x, batch_size=32)
    boxes = drop_overlapping_classes(boxes)
    for i, _boxes in enumerate(boxes):
        if i != 0:  # Don't predict bars for now
            if len(_boxes) > 0:
                for rowmin, colmin, rowmax, colmax in _boxes:
                    payload.append(
                        {
                            "value": {
                                "x": colmin / width * 100,
                                "y": rowmin / height * 100,
                                "width": (colmax - colmin) / width * 100,
                                "height": (rowmax - rowmin) / height * 100,
                                "rotation": 0,
                                "rectanglelabels": [variables["LENC"][i]],
                            },
                            "id": random_string(6),
                            "from_name": "tag",
                            "to_name": "img",
                            "type": "rectanglelabels",
                        }
                    )
    return payload


if __name__ == "__main__":
    vgg = load_model("vgg16-validated-five-classes.h5")
    x = imread("/tmp/medistrava.png")
    get_bboxes(x, vgg)

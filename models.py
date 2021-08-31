from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow_addons.image import connected_components
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from gramex.config import app_log
import pandas as pd


def _get_patch_sizes(height, width, min_patchsize=56, min_patches=2, keep=8):
    size = min(height, width)
    sizes = [
        min_patchsize * i for i in range(2, size // (min_patches * min_patchsize) + 1)
    ]
    final_sizes = {}
    for size in sizes:
        out_height = ceil((height - size + 1) / (size / 4))
        out_width = ceil((width - size + 1) / (size / 4))
        final_sizes[size] = (out_height, out_width)
    return pd.Series(final_sizes).drop_duplicates(keep="last").tail(keep).index


def draw_patches(x, **kwargs):
    _, nrows, ncols = x.shape[:3]
    figsize = kwargs.pop("figsize", (ncols, nrows))
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].imshow(x[0, i, j])
            ax[i, j].set_axis_off()


class Windowing(Layer):
    def __init__(self, stride, size, outsize=None, multires=False, min_patchsize=56):
        super(Windowing, self).__init__()
        self.size = size
        self.stride = stride
        self.outsize = outsize
        self.multires = multires
        self.min_patchsize = min_patchsize

    def _resize(self, patches, block_height, block_width, n_channels):
        patches = tf.reshape(
            patches,
            patches.shape[:3] + (block_height, block_width, n_channels),
        )
        if self.outsize is not None:
            batch, blockrow, blockcol = patches.shape[:3]
            patch_width, patch_height, n_channels = patches.shape[3:]
            resized = tf.image.resize(
                tf.reshape(
                    patches,
                    (
                        batch * blockrow * blockcol,
                        patch_width,
                        patch_height,
                        n_channels,
                    ),
                ),
                self.outsize,
            )
            patches = tf.reshape(
                resized, (batch, blockrow, blockcol) + self.outsize + (n_channels,)
            )
        return patches

    def call(self, inputs):
        height, width = inputs.shape[1:3]
        _, block_height, block_width, _ = self.size
        to_resize = False
        new_width = width
        if width % block_width != 0:
            new_width = block_width * (width // block_width + 1)
            to_resize = True
        new_height = height
        if height % block_height != 0:
            new_height = block_height * (height // block_height + 1)
            to_resize = True
        if to_resize:
            inputs = tf.image.resize(inputs, (new_height, new_width))
        if self.multires:
            tile_heights = tile_widths = _get_patch_sizes(
                new_height, new_width, self.min_patchsize
            )
            app_log.info(
                f"{len(tile_heights)} patches found for image of size ({height}, {width})"
            )
            with tf.device("cpu"):
                patches = [
                    tf.image.extract_patches(
                        inputs,
                        (1, h, w, 1),
                        (1, h / 4, w / 4, 1),
                        rates=[1, 1, 1, 1],
                        padding="VALID",
                    )
                    for h, w in zip(tile_heights, tile_widths)
                ]
                patches = [
                    self._resize(p, h, w, 3)
                    for p, h, w in zip(patches, tile_heights, tile_widths)
                ]
        else:
            patches = tf.image.extract_patches(
                inputs, self.size, self.stride, rates=[1, 1, 1, 1], padding="VALID"
            )
            patches = self._resize(patches, *self.size[1:-1], inputs.shape[-1])
        return patches


class WindowObjectDetector(Model):
    trainable = False

    def __init__(self, base, multires=False, bbox_threshold=0.5, min_patchsize=112):
        super(WindowObjectDetector, self).__init__()
        self.base = base
        self.bbox_threshold = bbox_threshold
        self.multires = multires
        self.min_patchsize = min_patchsize

    def predict(self, inputs, *args, **kwargs):
        block_height, block_width = self.base.input.shape[1:3]
        windowing = Windowing(
            [1, block_height / 4, block_width / 4, 1],
            [1, block_height, block_width, 1],
            multires=self.multires,
            min_patchsize=self.min_patchsize,
        )
        patches = windowing(inputs)
        if self.multires:
            preds = []
            for i, patch in enumerate(patches):
                batch, blockrow, blockcol = patch.shape[:3]
                patch = tf.reshape(
                    patch, (batch * blockrow * blockcol) + patch.shape[3:]
                )
                if patch.shape[1:-1] != (block_height, block_width):
                    patch = tf.image.resize(patch, (block_height, block_width))
                prob = tf.nn.softmax(self.base.predict(patch, *args, **kwargs), axis=1)
                prob = tf.reshape(prob, (1, blockrow, blockcol, prob.shape[-1]))
                if i > 0:
                    prob = tf.image.resize(prob, preds[0].shape[1:-1], "nearest")
                preds.append(prob)
            prob = tf.reduce_max(tf.stack(preds), axis=1)[0]
            blockrow, blockcol = patches[0].shape[1:3]
        else:
            batch, blockrow, blockcol = patches.shape[:3]
            patches = tf.reshape(
                patches, (batch * blockrow * blockcol,) + patches.shape[3:]
            )
            prob = tf.nn.softmax(self.base.predict(patches, *args, **kwargs), axis=1)
            prob = tf.reshape(prob, (blockrow, blockcol, prob.shape[-1]))
        mask = tf.transpose(tf.cast(prob > self.bbox_threshold, tf.int32))
        kernel = tf.zeros((3, 3, 1), tf.int32)
        dilated = tf.nn.dilation2d(
            tf.reshape(mask, mask.shape + (1,)),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 1, 1, 1),
            data_format="NHWC",
            padding="SAME",
        )[..., 0]
        labels = tf.transpose(connected_components(dilated))
        boxes = []
        for label in range(labels.shape[-1]):
            label_mask = labels[..., label]
            unique, _ = tf.unique(tf.reshape(label_mask, (blockrow * blockcol,)))
            _boxes = []
            for region in unique[unique > 0]:
                region_ix = tf.where(label_mask == region)
                row, col = tf.transpose(region_ix)
                rowmin = tf.reduce_min(row)
                colmin = tf.reduce_min(col)
                rowmax = tf.reduce_max(row)
                colmax = tf.reduce_max(col)
                _boxes.append([rowmin, colmin, rowmax, colmax])
            boxes.append(np.array(_boxes))
        return blockrow, blockcol, boxes


if __name__ == "__main__":
    model = WindowObjectDetector(
        tf.keras.models.load_model("checkpoints/resnet50-best.h5"), multires=False
    )
    x = tf.keras.applications.resnet50.preprocess_input(
        tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img("/tmp/test.png")
        )
    )
    x.shape = (1,) + x.shape
    brow, bcol, labels = model.predict(x)

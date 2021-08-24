from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow_addons.image import connected_components
import tensorflow as tf
from skimage.io import imread
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt


def draw_patches(x, **kwargs):
    _, nrows, ncols = x.shape[:3]
    figsize = kwargs.pop("figsize", (ncols, nrows))
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].imshow(x[0, i, j])
            ax[i, j].set_axis_off()


class Windowing(Layer):
    def __init__(self, stride, size, outsize=None, multires=False):
        super(Windowing, self).__init__()
        self.size = size
        self.stride = stride
        self.outsize = outsize
        self.multires = multires

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
            tile_heights = [
                block_height * i for i in range(1, height // block_height + 1)
            ]
            tile_widths = [block_width * i for i in range(1, width // block_width + 1)]
            with tf.device('cpu'):
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

    def __init__(self, base, multires=False, bbox_threshold=0.7):
        super(WindowObjectDetector, self).__init__()
        self.base = base
        self.bbox_threshold = bbox_threshold
        self.multires = multires

    def predict(self, inputs, *args, **kwargs):
        block_height, block_width = self.base.input.shape[1:3]
        windowing = Windowing(
            [1, block_height / 4, block_width / 4, 1], [1, block_height, block_width, 1]
        )
        patches = windowing(inputs)
        batch, blockrow, blockcol = patches.shape[:3]
        patches = tf.reshape(patches, (batch * blockrow * blockcol,) + patches.shape[3:])
        prob = tf.nn.softmax(self.base.predict(patches, *args, **kwargs), axis=1)
        prob = tf.reshape(prob, (blockrow, blockcol, prob.shape[-1]))
        mask = prob > self.bbox_threshold
        labels = tf.transpose(connected_components(tf.transpose(mask)))
        boxes = []
        for label in range(labels.shape[-1]):
            label_mask = labels[..., label]
            unique, _ = tf.unique(tf.reshape(label_mask, (blockrow * blockcol,)))
            _boxes = []
            for region in unique[unique > 0]:
                region_ix = tf.where(label_mask == region)
                row, col = tf.transpose(region_ix)
                rowmin = tf.reduce_min(row) - 1
                colmin = tf.reduce_min(col) - 1
                rowmax = tf.reduce_max(row) + 1
                colmax = tf.reduce_max(col) + 1
                _boxes.append([rowmin, colmin, rowmax, colmax])
            boxes.append(_boxes)
        return blockrow, blockcol, boxes


if __name__ == "__main__":
    x = imread("choropleth.png")
    x = rgba2rgb(x)
    x.shape = (1,) + x.shape
    window = Windowing(
        [1, 224 / 4, 224 / 4, 1], [1, 224, 224, 1], outsize=(224, 224), multires=True
    )
    patches = window(x)

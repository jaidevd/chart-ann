from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow_addons.image import connected_components
import tensorflow as tf
from skimage.io import imread
from skimage.color import rgba2rgb


class Windowing(Layer):
    def __init__(self, stride, size):
        super(Windowing, self).__init__()
        self.size = size
        self.stride = stride

    def call(self, inputs):
        _, block_height, block_width, _ = self.size
        height, width = inputs.shape[1:3]
        to_resize = False
        if width % block_width != 0:
            new_width = block_width * (width // block_width + 1)
            to_resize = True
        if height % block_height != 0:
            new_height = block_height * (height // block_height + 1)
            to_resize = True
        if to_resize:
            inputs = tf.image.resize(inputs, (new_height, new_width))
        patches = tf.image.extract_patches(
            inputs, self.size, self.stride, rates=[1, 1, 1, 1], padding='VALID')
        return patches
        # batch, row, col, patchsize = patches.shape
        # n_channels = inputs.shape[-1]
        # return tf.reshape(patches, (batch * row * col, block_height, block_width, n_channels))


class Morphology(Layer):
    """Morphology.
    """


class WindowObjectDetector(Model):
    trainable = False

    def __init__(self, base, bbox_threshold=0.7):
        super(WindowObjectDetector, self).__init__()
        self.base = base
        block_height, block_width = self.base.input.shape[1:3]
        self.windowing = Windowing([1, block_height / 4, block_width / 4, 1],
                                   [1, block_height, block_width, 1])
        self.bbox_threshold = bbox_threshold

    def predict(self, inputs, *args, **kwargs):
        patches = self.windowing(inputs)
        batch, row, col, patchsize = patches.shape
        n_channels = inputs.shape[-1]
        _, block_height, block_width, _ = self.windowing.size
        patches = tf.reshape(patches, (batch * row * col, block_height, block_width, n_channels))
        prob = tf.nn.softmax(self.base.predict(patches, *args, **kwargs), axis=1)
        prob = tf.reshape(prob, (row, col, prob.shape[-1]))
        mask = prob > self.bbox_threshold
        labels = tf.transpose(connected_components(tf.transpose(mask)))
        for i in range(labels.shape[-1]):
            loc = tf.where(labels[..., i])
            maxcol, maxrow = tf.reduce_max(loc, axis=0)
            mincol, minrow = tf.reduce_min(loc, axis=0)
            yield i, minrow, mincol, maxrow, maxcol


if __name__ == "__main__":
    base = load_model('vgg16-validated-five-classes.h5')
    model = WindowObjectDetector(base, bbox_threshold=0.7)

    x = imread('/tmp/cap-choropleth.png')
    x = rgba2rgb(x)
    x.shape = (1,) + x.shape
    prob = model.predict(x, batch_size=64)

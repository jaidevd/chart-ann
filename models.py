from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.io import imread
from skimage.color import rgba2rgb
from skimage.transform import resize


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

    def __init__(self, base):
        super(WindowObjectDetector, self).__init__()
        self.base = base
        block_height, block_width = self.base.input.shape[1:3]
        self.windowing = Windowing([1, block_height / 4, block_width / 4, 1],
                                   [1, block_height, block_width, 1])

    def predict(self, inputs, *args, **kwargs):
        patches = self.windowing(inputs)
        batch, row, col, patchsize = patches.shape
        n_channels = inputs.shape[-1]
        _, block_height, block_width, _ = self.windowing.size
        patches = tf.reshape(patches, (batch * row * col, block_height, block_width, n_channels))
        prob = tf.nn.softmax(self.base.predict(patches, *args, **kwargs), axis=1)
        return tf.reshape(prob, (row, col, prob.shape[-1]))


if __name__ == "__main__":
    base = load_model('vgg16-validated-five-classes.h5')
    model = WindowObjectDetector(base)

    x = imread('/tmp/choropleth.png')
    x = rgba2rgb(x)
    x = resize(x, (2240, 1120))  # NOQA: E912
    x.shape = (1,) + x.shape
    model.predict(x, batch_size=64)

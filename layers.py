from tensorflow.keras import layers
import tensorflow as tf


class Windowing(layers.Layer):
    def __init__(self, stride, size):
        self.size = size
        self.stride = stride

    def call(self, inputs):
        block_width, block_height = self.size
        width, height = inputs.shape[1:3]
        if width % block_width != 0:
            new_width = block_width * (width // block_width + 1)
        else:
            new_width = block_width
        if height % block_height != 0:
            new_height = block_height * (height // block_height + 1)
        else:
            new_height = block_height
        x = tf.image.resize(inputs, (new_width, new_height))
        patches = tf.image.extract_patches(
            x, self.size, self.stride, rates=[1, 1, 1, 1], padding='VALID')
        batch, row, col, patchsize = patches.shape
        n_channels = inputs.shape[-1]
        return tf.reshape(patches, (batch * row * col, block_height, block_width, n_channels))


class Morphology(layers.layer):
    """Morphology.
    """

import tensorflow as tf
from tensorflow import keras


class PaddedConv2D(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        padding=0,
        strides=1,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        self.conv2d = keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

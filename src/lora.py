import tensorflow as tf
from tensorflow import keras
from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)


class LoraInjectedLinearWrapper(keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        bias=False,
        r=4,
        dropout_p=0.0,
        scale=1.0,
        **kwargs,
    ):
        super().__init__(name=f"{original_layer.name}_lora", **kwargs)

        self.linear = original_layer
        self.linear.trainable = False
        self.config = self.linear.get_config()

        output_dim = self.config["units"]

        if r > output_dim:
            raise ValueError(f"LoRA rank {r} must be less or equal than {output_dim}")

        self.r = r
        self.bias = bias
        self.lora_down = keras.layers.Dense(
            r, use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform()
        )
        if dropout_p > 0:
            self.dropout = keras.layers.Dropout(dropout_p)
        else:
            self.dropout = lambda x: x
        self.lora_up = keras.layers.Dense(
            output_dim,
            input_shape=(r,),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Zeros(),
        )
        self.lora_down.trainable = True
        self.lora_up.trainable = True
        self.scale = scale

    def call(self, inputs):
        x = self.lora_down(inputs)
        x = self.lora_up(x)
        x = self.dropout(x)
        x = self.linear(inputs) + x * self.scale
        return x

    def get_lora(self):
        return tf.linalg.matmul(self.lora_down.weights, self.lora_up.weights)


class LoraInjectedConv2DWrapper(keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        r: int = 4,
        dropout_p: float = 0.0,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=f"{original_layer.name}_lora", **kwargs)

        self.conv = original_layer
        self.conv.trainable = False
        self.config = self.conv.get_config()

        if r > self.config["filters"]:
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {self.config['filters']}"
            )
        self.r = r

        self._lora_down_config = {
            **self.config,
            "filters": r,
            "use_bias": False,
            "trainable": True,
            "kernel_initializer": tf.keras.initializers.HeUniform(),
        }
        self._lora_up_config = {
            **self.config,
            "kernel_size": 1,
            "strides": 1,
            "padding": 0,
            "use_bias": False,
            "trainable": True,
            "kernel_initializer": tf.keras.initializers.Zeros(),
        }
        self.lora_down = PaddedConv2D(**self._lora_down_config)
        if dropout_p > 0:
            self.dropout = keras.layers.Dropout(dropout_p)
        else:
            self.dropout = lambda x: x
        self.lora_up = PaddedConv2D(**self._lora_up_config)
        self.scale = scale

    def call(self, inputs):
        x = self.lora_down(inputs)
        x = self.lora_up(x)
        x = self.dropout(x)
        x = self.conv(inputs) + x * self.scale
        return x

    def get_lora(self):
        return tf.linalg.matmul(self.lora_down.weights, self.lora_up.weights)

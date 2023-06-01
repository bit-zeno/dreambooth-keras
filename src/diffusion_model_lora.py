# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
from tensorflow import keras

from layers import PaddedConv2D
from lora import LoraInjectedLinearWrapper, LoraInjectedConv2DWrapper


class LoRADiffusionModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length, 768))
        t_embed_input = keras.layers.Input((320,))
        latent = keras.layers.Input((img_height // 8, img_width // 8, 4))

        t_emb = LoraInjectedLinearWrapper(keras.layers.Dense(1280), t_embed_input)(
            t_embed_input
        )
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = LoraInjectedLinearWrapper(keras.layers.Dense(1280), t_emb)(t_emb)

        # Downsampling flow

        outputs = []
        x = LoraInjectedConv2DWrapper(PaddedConv2D(320, kernel_size=3, padding=1))(
            latent
        )
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = LoraInjectedConv2DWrapper(PaddedConv2D(320, 3, strides=2, padding=1))(
            x
        )  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = LoraInjectedConv2DWrapper(PaddedConv2D(640, 3, strides=2, padding=1))(
            x
        )  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = LoraInjectedConv2DWrapper(PaddedConv2D(1280, 3, strides=2, padding=1))(
            x
        )  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = LoraInjectedConv2DWrapper(PaddedConv2D(4, kernel_size=3, padding=1))(x)

        super().__init__([latent, t_embed_input, context], output, name=name)

        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",  # noqa: E501
                file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",  # noqa: E501
            )
            self.load_weights(diffusion_model_weights_fpath)


class ResBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.entry_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            lambda x: LoraInjectedLinearWrapper(keras.layers.Dense(output_dim), x),
        ]
        self.exit_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            LoraInjectedConv2DWrapper(PaddedConv2D(output_dim, 3, padding=1)),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = LoraInjectedConv2DWrapper(
                PaddedConv2D(self.output_dim, 1)
            )
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        x = PaddedConv2D(self.output_dim, 3, padding=1)(x)
        for i, layer in enumerate(self.embedding_flow):
            if i == len(self.embedding_flow) - 1:
                embeddings = layer(embeddings)(embeddings)
            else:
                embeddings = layer(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        channels = num_heads * head_size
        self.fully_connected = fully_connected
        if fully_connected:
            self.proj1 = lambda x: LoraInjectedLinearWrapper(
                keras.layers.Dense(num_heads * head_size), x
            )
        else:
            self.proj1 = LoraInjectedConv2DWrapper(
                PaddedConv2D(num_heads * head_size, 1)
            )
        self.transformer_block = BasicTransformerBlock(channels, num_heads, head_size)
        if fully_connected:
            self.proj2 = lambda x: LoraInjectedLinearWrapper(
                keras.layers.Dense(channels), x
            )
        else:
            self.proj2 = LoraInjectedConv2DWrapper(PaddedConv2D(channels, 1))

    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        if self.fully_connected:
            x = self.proj1(x)(x)
        else:
            x = self.proj1(x)
        x = tf.reshape(x, (-1, h * w, c))
        x = self.transformer_block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        if self.fully_connected:
            return self.proj2(x)(x) + inputs
        else:
            return self.proj2(x) + inputs


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = lambda x: LoraInjectedLinearWrapper(keras.layers.Dense(dim), x)

    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1([self.norm1(inputs), None]) + inputs
        x = self.attn2([self.norm2(x), context]) + x
        _tmp = self.geglu(self.norm3(x))
        return self.dense(_tmp)(_tmp) + x


class CrossAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = lambda x: LoraInjectedLinearWrapper(
            keras.layers.Dense(num_heads * head_size, use_bias=False), x
        )
        self.to_k = lambda x: LoraInjectedLinearWrapper(
            keras.layers.Dense(num_heads * head_size, use_bias=False), x
        )
        self.to_v = lambda x: LoraInjectedLinearWrapper(
            keras.layers.Dense(num_heads * head_size, use_bias=False), x
        )
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = lambda x: LoraInjectedLinearWrapper(
            keras.layers.Dense(num_heads * head_size), x
        )

    def call(self, inputs):
        inputs, context = inputs
        context = inputs if context is None else context
        q, k, v = (
            self.to_q(inputs)(inputs),
            self.to_k(context)(context),
            self.to_v(context)(context),
        )
        q = tf.reshape(q, (-1, inputs.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attn = td_dot(weights, v)
        attn = tf.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
        out = tf.reshape(attn, (-1, inputs.shape[1], self.num_heads * self.head_size))
        return self.out_proj(out)(out)


class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = keras.layers.UpSampling2D(2)
        self.conv = LoraInjectedConv2DWrapper(PaddedConv2D(channels, 3, padding=1))

    def call(self, inputs):
        x = self.ups(inputs)
        return self.conv(x)


class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = lambda x: LoraInjectedLinearWrapper(
            keras.layers.Dense(output_dim * 2), x
        )

    def call(self, inputs):
        x = self.dense(inputs)(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)


def td_dot(a, b):
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))

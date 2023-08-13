import tensorflow as tf
from metadata import PAD, XY_POINT_LANDMARKS
from itertools import groupby
import numpy as np

class LandmarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(LandmarkEmbedding, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(d_model, use_bias=False, name='proj')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")

    def call(self, x, training):
        x = self.dense(x)
        x = self.norm(x, training=training)
        return x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout, max_length, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.scale = self.d_model ** -0.5
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_length = max_length

        self.qkv = tf.keras.layers.Dense(3 * d_model, use_bias=False, name="qkv")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(d_model, use_bias=False, name="out")
        self.rel_pos_emb = tf.keras.layers.Embedding(2 * max_length + 1, self.head_dim)

    def call(self, inputs, mask, training):
        seq_length = tf.shape(inputs)[1]
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3)) \
            (tf.keras.layers.Reshape((-1, self.num_heads, self.d_model * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.d_model // self.num_heads] * 3, axis=-1)  # B x #head x
        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        seq = tf.range(seq_length)
        dist = tf.expand_dims(seq, 1) - tf.expand_dims(seq, 0)
        dist = (
                tf.clip_by_value(
                    dist, clip_value_min=-self.max_length, clip_value_max=self.max_length
                )
                + self.max_length
        )
        rel_pos_emb = self.rel_pos_emb(dist)
        pos_attn = tf.einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        attn = attn + pos_attn

        mask = mask[:, None, None, :]
        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.dropout(attn, training=training)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.d_model))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


class Swish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)


class GLU(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs):
        out, gate = tf.split(inputs, 2, axis=self.dim)
        return out * tf.sigmoid(gate)


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, dilation_rate, use_bias, depthwise_initializer='glorot_uniform', **kwargs):
        super(CausalDWConv1D, self).__init__(**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name="causal_pad")
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size,
            strides=1,
            dilation_rate=dilation_rate,
            padding='valid',
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name="causal_conv1d")

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x


class ConformerFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim, dropout, **kwargs):
        super(ConformerFeedForward, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(mlp_dim, activation=Swish(), name="dense1")
        self.dropout1 = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1), name="dropout1")
        self.dense2 = tf.keras.layers.Dense(d_model, name="dense2")
        self.dropout2 = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1), name="dropout2")

    def call(self, inputs, training):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x


class ConvModule(tf.keras.layers.Layer):
    def __init__(self, d_model, conv_dim, kernel_size, dilation_rate, dropout, **kwargs):
        super(ConvModule, self).__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")
        self.conv1d1 = tf.keras.layers.Conv1D(filters=conv_dim * 2, kernel_size=1, use_bias=False, name="conv1d1")
        self.act1 = GLU(dim=-1, name="glu")
        self.causal_conv = CausalDWConv1D(kernel_size, dilation_rate, use_bias=False, name="causal_conv")
        self.norm2 = tf.keras.layers.BatchNormalization(momentum=0.95, name="bn")
        self.act2 = Swish(name="swish")
        self.conv1d2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, use_bias=False, name="conv1d2")
        self.dropout = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1), name="dropout")

    def call(self, inputs, training):
        x = self.norm1(inputs, training=training)
        x = self.conv1d1(x)
        x = self.act1(x)
        x = self.causal_conv(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.conv1d2(x)
        x = self.dropout(x, training=training)
        return x


class ConformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            d_model,
            num_heads,
            conv_dim,
            mlp_dim,
            kernel_size,
            dilation_rate,
            attn_dropout,
            hidden_dropout,
            max_length,
            **kwargs
    ):
        super(ConformerEncoderLayer, self).__init__(**kwargs)
        self.ff1_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ff1_ln")
        self.ff1 = ConformerFeedForward(
            d_model=d_model,
            mlp_dim=mlp_dim,
            dropout=hidden_dropout,
            name="ff1")

        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="attn_ln")
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            max_length=max_length,
            name="self_attn")
        self.attn_dropout = tf.keras.layers.Dropout(hidden_dropout, noise_shape=(None, 1, 1), name="attn_dropout")

        self.conv = ConvModule(
            d_model=d_model,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=hidden_dropout,
            name="conv_module")

        self.ff2_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ff2_ln")
        self.ff2 = ConformerFeedForward(
            d_model=d_model,
            mlp_dim=mlp_dim,
            dropout=hidden_dropout,
            name="ff2")
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")

    def call(self, inputs, mask, training):
        ff1_out = self.ff1_norm(inputs, training=training)
        ff1_out = self.ff1(ff1_out, training=training)
        ff1_out = 0.5 * ff1_out + inputs

        attn_out = self.attn_norm(ff1_out, training=training)
        attn_out = self.attn(attn_out, mask=mask, training=training)
        attn_out = self.attn_dropout(attn_out, training=training)
        attn_out = attn_out + ff1_out

        conv_out = self.conv(attn_out, training=training)
        conv_out = conv_out + attn_out

        ff2_out = self.ff2_norm(conv_out, training=training)
        ff2_out = self.ff2(ff2_out, training=training)
        ff2_out = 0.5 * ff2_out + conv_out
        ff2_out = self.final_norm(ff2_out, training=training)
        return ff2_out


class ConformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, kernel_size, dilation_rate, num_heads, mlp_dim, conv_dim, attn_dropout,
                 hidden_dropout, max_source_length, **kwargs):
        super(ConformerEncoder, self).__init__(**kwargs)
        self.embedding = LandmarkEmbedding(d_model, name="landmark_embedding")
        self.layers = [
            ConformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                conv_dim=conv_dim,
                mlp_dim=mlp_dim,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                attn_dropout=attn_dropout,
                hidden_dropout=hidden_dropout,
                max_length=max_source_length,
                name=f"encoder_layer{i}") for i in range(num_layers)]

    def call(self, inputs, mask, training):
        x = self.embedding(inputs, training=training)
        for layer in self.layers:
            x = layer(x, mask=mask, training=training)
        return x


def get_initializer(initializer_range: float = 0.02) -> tf.keras.initializers.TruncatedNormal:
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class LMHead(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, activation, **kwargs):
        super(LMHead, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(
            2 * d_model,
            kernel_initializer=get_initializer(),
            activation='relu',
            use_bias=True,
            name="dense1")
        #self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        self.dense2 = tf.keras.layers.Dense(
            vocab_size,
            kernel_initializer=get_initializer(),
            use_bias=True,
            name="dense2")

    def call(self, x):
        x = self.dense1(x)
        #x = self.layer_norm(x)
        return self.dense2(x)


class Conformer(tf.keras.Model):
    def __init__(
        self,
        num_encoder_layers: int,
        encoder_hidden_dim: int,
        encoder_mlp_dim: int,
        encoder_conv_dim: int,
        encoder_kernel_size: int,
        encoder_dilation_rate: int,
        encoder_num_heads: int,
        max_source_length: int,
        vocab_size: int,
        pad_token_id: int,
        attn_dropout: float,
        hidden_dropout: float,
        activation: str):
        super(Conformer, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.encoder = ConformerEncoder(
            num_layers=num_encoder_layers,
            d_model=encoder_hidden_dim,
            kernel_size=encoder_kernel_size,
            dilation_rate=encoder_dilation_rate,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            conv_dim=encoder_conv_dim,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            max_source_length=max_source_length,
            name="encoder")

        self.ctc_head = LMHead(
            d_model=encoder_hidden_dim,
            vocab_size=vocab_size,
            activation=activation,
            name="ctc_head")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        source = inputs
        encoder_attention_mask = tf.reduce_sum(tf.cast(source!=PAD, tf.float32), axis=2)!=0
        encoder_out = self.encoder(source, mask=encoder_attention_mask, training=training)
        return self.ctc_head(encoder_out)

    @property
    def metrics(self):
        return [self.loss_metric]

    def compile(self, optimizer, loss_fn):
        super().compile(optimizer=optimizer)
        self.loss_fn = loss_fn

    def train_step(self, batch):
        source = batch[0]
        target = batch[1]

        logits_mask = tf.reduce_sum(tf.cast(source != PAD, tf.float32), axis=2) != 0
        logits_length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        target_length = tf.reduce_sum(tf.cast(target != self.pad_token_id, tf.int32), axis=-1)

        with tf.GradientTape() as tape:
            ctc_out = self(source, training=True)
            loss = self.loss_fn(
                labels=target,
                logits=ctc_out,
                label_length=target_length,
                logits_length=logits_length,
                blank_idx=self.pad_token_id,
                logits_time_major=False)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch[0]
        target = batch[1]
        logits_mask = tf.reduce_sum(tf.cast(source != PAD, tf.float32), axis=2) != 0
        logits_length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        target_length = tf.reduce_sum(tf.cast(target != self.pad_token_id, tf.int32), axis=-1)
        ctc_out = self(source, training=False)
        loss = self.loss_fn(
            labels=target,
            logits=ctc_out,
            label_length=target_length,
            logits_length=logits_length,
            blank_idx=self.pad_token_id,
            logits_time_major=False)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def batch_generate(self, source):
        logits_mask = tf.reduce_sum(tf.cast(source != PAD, tf.float32), axis=2) != 0
        lengths = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        logits = self(source, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return preds, lengths


class TFLiteModel(tf.Module):
    def __init__(self, model, preprocess_layer, pad_token_id, max_gen_length):
        super(TFLiteModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.max_gen_length = max_gen_length
        self.model = model
        self.preprocess_layer = preprocess_layer

    @tf.function(jit_compile=True)
    def encoder(self, x):
        logits_mask = tf.reduce_sum(tf.cast(x != PAD, tf.float32), axis=2) != 0
        length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        encoder_out = self.model.encoder(x, mask=logits_mask, training=False)
        logits = self.model.ctc_head(encoder_out)
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return pred, length

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(XY_POINT_LANDMARKS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(XY_POINT_LANDMARKS))), lambda: tf.identity(x))
        x = x[0]

        x = self.preprocess_layer(x)
        x = x[None]
        x, length = self.encoder(x)
        x = x[0]

        diff = tf.not_equal(x[:-1], x[1:]) #TODO: check
        adjacent_indices = tf.where(diff)[:, 0]
        x = tf.gather(x, adjacent_indices)
        mask = x != self.pad_token_id
        x = tf.boolean_mask(x, mask, axis=0)

        x = x[:self.max_gen_length]
        x = tf.one_hot(x, 59) # how about not in 59?
        return {'outputs': x}


class TFLiteModelBestPath(tf.Module):
    def __init__(self, model, preprocess_layer, pad_token_id, max_gen_length):
        super(TFLiteModelBestPath, self).__init__()
        self.pad_token_id = pad_token_id
        self.max_gen_length = max_gen_length
        self.model = model
        self.preprocess_layer = preprocess_layer

    @tf.function(jit_compile=True)
    def encoder(self, x):
        logits_mask = tf.reduce_sum(tf.cast(x != PAD, tf.float32), axis=2) != 0
        length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        encoder_out = self.model.encoder(x, mask=logits_mask, training=False)
        logits = self.model.ctc_head(encoder_out)
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return pred, length

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(XY_POINT_LANDMARKS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(XY_POINT_LANDMARKS))), lambda: tf.identity(x))
        x = x[0]

        x = self.preprocess_layer(x)
        x = x[None]
        x, length = self.encoder(x)
        x = x[0]

        shifted_x = tf.roll(x, shift=-1, axis=0)
        is_same_as_next = tf.math.equal(x[:-1], shifted_x[:-1])

        # Add a 'False' to the end to keep the last element
        is_same_as_next = tf.concat([is_same_as_next, [False]], axis=0)

        # Filter out elements that are duplicates or equal to pad_token_id
        x_deduplicated = tf.boolean_mask(x, tf.math.logical_not(is_same_as_next))
        x = tf.boolean_mask(x_deduplicated, tf.math.not_equal(x_deduplicated, self.pad_token_id))

        x = x[:self.max_gen_length]
        x = tf.one_hot(x, 59) # how about not in 59?
        return {'outputs': x}


class TFLiteModelBeamSearch(tf.Module):
    def __init__(self, model, preprocess_layer, pad_token_id, max_gen_length):
        super(TFLiteModelBeamSearch, self).__init__()
        self.pad_token_id = pad_token_id
        self.max_gen_length = max_gen_length
        self.model = model
        self.preprocess_layer = preprocess_layer

    @tf.function(jit_compile=True)
    def encoder(self, x):
        logits_mask = tf.reduce_sum(tf.cast(x != PAD, tf.float32), axis=2) != 0
        length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        encoder_out = self.model.encoder(x, mask=logits_mask, training=False)
        logits = self.model.ctc_head(encoder_out)
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return pred, length

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(XY_POINT_LANDMARKS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(XY_POINT_LANDMARKS))), lambda: tf.identity(x))
        x = x[0]

        x = self.preprocess_layer(x)
        x = x[None]
        x, length = self.encoder(x)
        x = x[0]

        #length = [len(x)]
        #x = tf.expand_dims(x, axis=1)
        #x = tf.nn.ctc_greedy_decoder(x, length, merge_repeated=True, blank_index=self.pad_token_id)[0][0].values


        shifted_x = tf.concat([x[1:], x[:1]], axis=0) #tf.roll(x, shift=-1, axis=0)
        is_same_as_next = tf.math.equal(x[:-1], shifted_x[:-1])

        # Add a 'False' to the end to keep the last element
        is_same_as_next = tf.concat([is_same_as_next, [False]], axis=0)

        # Filter out elements that are duplicates or equal to pad_token_id
        x_deduplicated = tf.boolean_mask(x, tf.math.logical_not(is_same_as_next))
        x = tf.boolean_mask(x_deduplicated, tf.math.not_equal(x_deduplicated, self.pad_token_id))


        x = x[:self.max_gen_length]
        x = tf.one_hot(x, 59) # how about not in 59?
        return {'outputs': x}
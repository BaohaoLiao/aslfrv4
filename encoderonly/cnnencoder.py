import tensorflow as tf
import numpy as np
from metadata import PAD, XY_POINT_LANDMARKS


class LandmarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(LandmarkEmbedding, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(d_model, use_bias=False, name='proj')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")

    def call(self, x):
        x = self.dense(x)
        x = self.norm(x)
        return x


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


class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False, name="conv1d")

    def call(self, inputs, mask):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class Conv1DBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, conv_dim, kernel_size, dilation_rate, dropout, activation, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(conv_dim, use_bias=True, activation=activation, name="dense1")
        self.causal_conv = CausalDWConv1D(
            kernel_size=kernel_size, dilation_rate=dilation_rate, use_bias=False, name="causal_conv")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")
        self.eca = ECA(name="eca")
        self.dense2 = tf.keras.layers.Dense(d_model, use_bias=True, name="dense2")
        self.dropout = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1), name="dropout")

    def call(self, x, mask, training):
        skip = x
        x = self.dense1(x)
        x = self.causal_conv(x)
        x = self.norm(x)
        x = self.eca(x, mask=mask)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = tf.keras.layers.add([x, skip])
        return x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.scale = self.d_model ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * d_model, use_bias=False, name="qkv")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(d_model, use_bias=False, name="out")

    def call(self, inputs, mask, training):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))\
            (tf.keras.layers.Reshape((-1, self.num_heads, self.d_model * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.d_model // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        mask = mask[:, None, None, :]
        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.dropout(attn, training=training)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.d_model))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim, activation, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.supports_masking = True
        self.dense1 = tf.keras.layers.Dense(
            mlp_dim,
            activation=activation,
            name="dense1")
        self.dense2 = tf.keras.layers.Dense(d_model, name="dense2")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim, attn_dropout, hidden_dropout, activation, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.attention = MultiHeadSelfAttention(d_model, num_heads, attn_dropout, name="custom_attention")
        self.attn_dropout = tf.keras.layers.Dropout(hidden_dropout, noise_shape=(None, 1, 1))

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.mlp = FeedForward(d_model, mlp_dim, activation, name="mlp")
        self.mlp_dropout = tf.keras.layers.Dropout(hidden_dropout, noise_shape=(None, 1, 1))

    def call(self, inputs, mask, training):
        x = inputs
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        x = self.attn_dropout(x, training=training)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.mlp_dropout(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x


class CNNEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, dilation_rate, num_heads, mlp_dim, conv_dim, attn_dropout, hidden_dropout,
        activation, **kwargs):
        super(CNNEncoderLayer, self).__init__(**kwargs)
        self.convs = [Conv1DBlock(
            d_model=d_model,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=hidden_dropout,
            activation=activation,
            name=f"conv{i}") for i in range(3)]
        self.transformer_block = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            name="transformer_encoder")

    def call(self, inputs, mask, training):
        x = inputs
        for conv in self.convs:
            x = conv(x, mask=mask, training=training)
        x = self.transformer_block(x, mask=mask, training=training)
        return x


class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, kernel_size, dilation_rate, num_heads, mlp_dim, conv_dim, attn_dropout,
            hidden_dropout, activation, **kwargs):
        super(CNNEncoder, self).__init__(**kwargs)
        self.embedding = LandmarkEmbedding(d_model, name="landmark_embedding")
        self.layers = [
            CNNEncoderLayer(
                d_model=d_model,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                conv_dim=conv_dim,
                attn_dropout=attn_dropout,
                hidden_dropout=hidden_dropout,
                activation=activation,
                name=f"encoder_layer{i}") for i in range(num_layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")

    def call(self, inputs, mask, training):
        x = self.embedding(inputs)
        for layer in self.layers:
            x = layer(x, mask=mask, training=training)
        x = self.norm(x)
        return x


class LMHead(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, activation, **kwargs):
        super(LMHead, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(
            d_model,
            activation=activation,
            use_bias=True,
            name="dense1")
        self.dense2 = tf.keras.layers.Dense(
            vocab_size,
            use_bias=True,
            name="dense2")

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)


class CNN(tf.keras.Model):
    def __init__(
        self,
        num_encoder_layers: int,
        encoder_hidden_dim: int,
        encoder_mlp_dim: int,
        encoder_conv_dim: int,
        encoder_kernel_size: int,
        encoder_dilation_rate: int,
        encoder_num_heads: int,
        vocab_size: int,
        attn_dropout: float,
        hidden_dropout: float,
        activation: str,
        pad_token_id: int, ):
        super(CNN, self).__init__()
        self.pad_token_id = pad_token_id
        self.encoder = CNNEncoder(
            num_layers=num_encoder_layers,
            d_model=encoder_hidden_dim,
            kernel_size=encoder_kernel_size,
            dilation_rate=encoder_dilation_rate,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            conv_dim=encoder_conv_dim,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            name="encoder")
        self.lm_head = LMHead(
            d_model=encoder_hidden_dim,
            vocab_size=vocab_size,
            activation=activation,
            name="lm_head")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        source = inputs
        attention_mask = tf.reduce_sum(tf.cast(source!=PAD, tf.float32), axis=2)!=0
        encoder_out = self.encoder(source, mask=attention_mask, training=training)
        return self.lm_head(encoder_out)

    @property
    def metrics(self):
        return [self.loss_metric]

    def compile(self, optimizer, loss_fn):
        super().compile(optimizer=optimizer)
        self.loss_fn = loss_fn

    def train_step(self, batch):
        source = batch[0]
        target = batch[1]
        logits_mask = tf.reduce_sum(tf.cast(source!=PAD, tf.float32), axis=2)!=0
        logits_length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        target_length = tf.reduce_sum(tf.cast(target != self.pad_token_id, tf.int32), axis=-1)

        with tf.GradientTape() as tape:
            preds = self(source, training=True)
            loss = self.loss_fn(
                labels=target,
                logits=preds,
                label_length=target_length,
                logits_length=logits_length,
                blank_idx=self.pad_token_id,
                logits_time_major=False
            )

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
        preds = self(source, training=True)
        loss = self.loss_fn(
            labels=target,
            logits=preds,
            label_length=target_length,
            logits_length=logits_length,
            blank_idx=self.pad_token_id,
            logits_time_major=False
        )
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def batch_generate(self, source):
        logits_mask = tf.reduce_sum(tf.cast(source != PAD, tf.float32), axis=2) != 0
        lengths = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        logits = self(source, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        tf.print(preds, lengths)
        return preds, lengths



class TFLiteModel(tf.Module):
    def __init__(self, model, preprocess_layer, start_token_id, end_token_id, pad_token_id, max_gen_length):
        super(TFLiteModel, self).__init__()
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
        self.max_gen_length = max_gen_length
        self.model = model
        self.preprocess_layer = preprocess_layer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(XY_POINT_LANDMARKS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(XY_POINT_LANDMARKS))), lambda: tf.identity(x))
        x = x[0]

        x = self.preprocess_layer(x)
        x = x[None]
        batch_size = tf.shape(x)[0]
        encoder_out, encoder_attention_mask = self.encoder(x)
        dec_input = tf.ones((batch_size, 1), dtype=tf.int32) * self.start_token_id

        for _ in tf.range(self.max_gen_length-1):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(dec_input, tf.TensorShape([1, None]))])
            logits = self.decoder(
                dec_input=dec_input,
                encoder_out=encoder_out,
                encoder_attention_mask=encoder_attention_mask)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
            if (last_logit == self.end_token_id):
                break
        x = dec_input[0]
        idx = tf.argmax(tf.cast(tf.equal(x, self.end_token_id), tf.int32))  #TODO: CHECK
        idx = tf.where(tf.math.less(idx, 1), tf.constant(2, dtype=tf.int64), idx)
        x = x[1:idx] # replace pad token?
        x = tf.one_hot(x, 59) # how about not in 59?
        return {'outputs': x}
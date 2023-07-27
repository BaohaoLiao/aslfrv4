import tensorflow as tf
import numpy as np
from metadata import PAD, XY_POINT_LANDMARKS


def get_initializer(initializer_range: float = 0.02) -> tf.keras.initializers.TruncatedNormal:
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

def _expand_mask(mask: tf.Tensor, tgt_len: int = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, tgt_seq_len, src_seq_len]`.
    """
    src_len = tf.shape(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = tf.tile(mask[:, None, :], (1, tgt_len, 1))
    return expanded_mask


class LandmarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_source_length, dropout, learnable_position, **kwargs):
        super(LandmarkEmbedding, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(d_model, use_bias=False, name='proj')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln")
        self.learnable_position = learnable_position
        if learnable_position:
            self.position_emb = self.add_weight(
                "pos_embeddings",
                shape=[max_source_length, d_model],
                initializer=get_initializer(),
            )
        else:
            self.position_emb = positional_encoding(max_source_length, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        length = tf.shape(x)[1]
        x = self.dense(x)
        if self.learnable_position:
            position_ids = tf.expand_dims(tf.range(start=0, limit=length, delta=1), axis=0)
            position_embs = tf.gather(params=self.position_emb, indices=position_ids)
        else:
            position_embs = self.position_emb[tf.newaxis, :length, :]
        x = x + position_embs
        x = self.norm(x)
        x = self.dropout(x, training=training)
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
    def __init__(self, num_layers, d_model, max_source_length, kernel_size, dilation_rate, num_heads, mlp_dim, conv_dim,
                 emb_dropout, attn_dropout, hidden_dropout, activation, learnable_position, **kwargs):
        super(CNNEncoder, self).__init__(**kwargs)
        self.embedding = LandmarkEmbedding(
            d_model=d_model,
            max_source_length=max_source_length,
            dropout=emb_dropout,
            learnable_position=learnable_position,
            name="landmark_embedding")

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


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_target_length, d_model, dropout, learnable_position, **kwargs):
        super(TokenEmbedding, self).__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            embeddings_initializer=get_initializer(),
            name="token_embedding",
        )
        self.learnable_position = learnable_position
        if learnable_position:
            self.position_emb = self.add_weight(
                "pos_embeddings",
                shape=[max_target_length, d_model],
                initializer=get_initializer(),
            )
        else:
            self.position_emb = positional_encoding(max_target_length, d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="emb_layernorm")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, input_ids, training):
        length = tf.shape(input_ids)[1]
        x = self.token_embeddings(input_ids)
        if self.learnable_position:
            position_ids = tf.expand_dims(tf.range(start=0, limit=length, delta=1), axis=0)
            position_embs = tf.gather(params=self.position_emb, indices=position_ids)
        else:
            position_embs = self.position_emb[tf.newaxis, :length, :]
        x = x + position_embs
        x = self.layernorm(x)
        x = self.dropout(x, training=training)
        return x


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim, num_heads, attn_dropout, hidden_dropout, prenorm, activation, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.prenorm = prenorm
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attn_dropout,
            name="self_attn")
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attn_dropout,
            name="encoder_attn")
        self.mlp = FeedForward(
            d_model=d_model,
            mlp_dim=mlp_dim,
            activation=activation,
            name="mlp")

        self.self_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="self_attn_norm")
        self.cross_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="cross_attn_norm")
        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_norm")

        self.self_dropout = tf.keras.layers.Dropout(hidden_dropout)
        self.cross_dropout = tf.keras.layers.Dropout(hidden_dropout)
        self.mlp_dropout = tf.keras.layers.Dropout(hidden_dropout)

    def call(self, x, encoder_out, attention_mask, encoder_attention_mask, training):
        if self.prenorm:
            self_attn_out = self.self_attn_norm(x)
            self_attn_out = self.self_attention(
                query=self_attn_out,
                value=self_attn_out,
                attention_mask=attention_mask,
                training=training,
                use_causal_mask=True)
            self_attn_out = self.self_dropout(self_attn_out, training=training)
            self_attn_out = x + self_attn_out

            cross_attn_out = self.cross_attn_norm(self_attn_out)
            cross_attn_out = self.cross_attention(
                query=cross_attn_out,
                value=encoder_out,
                attention_mask=encoder_attention_mask,
                training=training)
            cross_attn_out = self.cross_dropout(cross_attn_out, training=training)
            cross_attn_out = self_attn_out + cross_attn_out

            mlp_out = self.mlp_norm(cross_attn_out)
            mlp_out = self.mlp(mlp_out)
            mlp_out = self.mlp_dropout(mlp_out, training=training)
            mlp_out = cross_attn_out + mlp_out
        else:
            self_attn_out = self.self_attention(
                query=x,
                value=x,
                attention_mask=attention_mask,
                training=training,
                use_causal_mask=True)
            self_attn_out = self.self_dropout(self_attn_out, training=training)
            self_attn_out = self.self_attn_norm(self_attn_out + x)

            cross_attn_out = self.cross_attention(
                query=self_attn_out,
                value=encoder_out,
                attention_mask=encoder_attention_mask,
                training=training)
            cross_attn_out = self.cross_dropout(cross_attn_out, training=training)
            cross_attn_out = self.self_attn_norm(cross_attn_out + self_attn_out)

            mlp_out = self.mlp(cross_attn_out)
            mlp_out = self.mlp_dropout(mlp_out, training=training)
            mlp_out = self.mlp_norm(mlp_out + cross_attn_out)
        return mlp_out


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers, d_model, max_target_length, learnable_position, num_heads, mlp_dim,
        emb_dropout, attn_dropout, hidden_dropout, pad_token_id, prenorm, activation, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            max_target_length=max_target_length,
            d_model=d_model,
            dropout=emb_dropout,
            learnable_position=learnable_position,
            name="token_embedding")
        self.layers = [
            TransformerDecoderLayer(
                d_model=d_model,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                hidden_dropout=hidden_dropout,
                prenorm=prenorm,
                activation=activation,
                name=f"decoder_layers.{i}") for i in range(num_layers)]
        self.final_norm = None
        if prenorm:
            self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")

    def call(self, input_ids, encoder_out, encoder_attention_mask, training):
        seq_length = tf.shape(input_ids)[1]
        x = self.embedding(input_ids, training=training)

        attention_mask = tf.math.not_equal(input_ids, self.pad_token_id)
        extended_attention_mask = _expand_mask(attention_mask)
        encoder_extended_attention_mask = _expand_mask(encoder_attention_mask, seq_length)

        for _, decoder_layer in enumerate(self.layers):
            x = decoder_layer(
                x,
                encoder_out=encoder_out,
                attention_mask=extended_attention_mask,
                encoder_attention_mask=encoder_extended_attention_mask,
                training=training)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x

class LMHead(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, activation, **kwargs):
        super(LMHead, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=get_initializer(),
            activation=activation,
            use_bias=True,
            name="dense1")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        self.dense2 = tf.keras.layers.Dense(
            vocab_size,
            kernel_initializer=get_initializer(),
            use_bias=True,
            name="dense2")

    def call(self, x):
        x = self.dense1(x)
        x = self.layer_norm(x)
        return self.dense2(x)


class CNNEncoderv2TransformerDecoder(tf.keras.Model):
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
        num_decoder_layers: int,
        vocab_size: int,
        decoder_hidden_dim: int,
        decoder_mlp_dim: int,
        decoder_num_heads: int,
        max_target_length: int,
        pad_token_id: int,
        emb_dropout: float,
        attn_dropout: float,
        hidden_dropout: float,
        learnable_position: bool,
        prenorm: bool,
        activation: str):
        super(CNNEncoderv2TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.encoder = CNNEncoder(
            num_layers=num_encoder_layers,
            d_model=encoder_hidden_dim,
            max_source_length=max_source_length,
            kernel_size=encoder_kernel_size,
            dilation_rate=encoder_dilation_rate,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            conv_dim=encoder_conv_dim,
            emb_dropout=emb_dropout,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            learnable_position=learnable_position,
            name="encoder")
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            num_layers=num_decoder_layers,
            d_model=decoder_hidden_dim,
            max_target_length=max_target_length,
            learnable_position=learnable_position,
            num_heads=decoder_num_heads,
            mlp_dim=decoder_mlp_dim,
            emb_dropout=emb_dropout,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            pad_token_id=pad_token_id,
            prenorm=prenorm,
            activation=activation,
            name="decoder")

        self.encoder_proj = None
        if encoder_hidden_dim != decoder_hidden_dim:
            self.encoder_proj = tf.keras.layers.Dense(
                decoder_hidden_dim, use_bias=False, name="encoder_proj")
        self.lm_head = LMHead(
            d_model=decoder_hidden_dim,
            vocab_size=vocab_size,
            activation=activation,
            name="lm_head")

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.top1_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1_acc")
        self.top5_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")

    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]

        encoder_attention_mask = tf.reduce_sum(tf.cast(source!=PAD, tf.float32), axis=2)!=0
        encoder_out = self.encoder(source, mask=encoder_attention_mask, training=training)
        if self.encoder_proj is not None:
            encoder_out = self.encoder_proj(encoder_out)

        decoder_out = self.decoder(
            input_ids=target,
            encoder_out=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
            training=training
        )
        return self.lm_head(decoder_out)

    @property
    def metrics(self):
        return [self.loss_metric]

    def compile(self, optimizer, loss_fn):
        super().compile(optimizer=optimizer)
        self.loss_fn = loss_fn

    def train_step(self, batch):
        source = batch[0]
        masked_target = batch[1]
        target = batch[2]

        dec_input = masked_target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input], training=True)
            one_hot = tf.one_hot(dec_target, depth=self.vocab_size)
            mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_id))
            loss = self.loss_fn(one_hot, preds, sample_weight=mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_metric.update_state(loss)
        self.top1_acc_metric.update_state(one_hot, preds, sample_weight=mask)
        self.top5_acc_metric.update_state(one_hot, preds, sample_weight=mask)
        return {
            "loss": self.loss_metric.result(),
            "top1_acc": self.top1_acc_metric.result(),
            "top5_acc": self.top5_acc_metric.result()}

    def test_step(self, batch):
        source = batch[0]
        target = batch[1]
        assert batch[1] == batch[2]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input], training=False)
        one_hot = tf.one_hot(dec_target, depth=self.vocab_size)
        mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_id))
        loss = self.loss_fn(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        self.top1_acc_metric.update_state(one_hot, preds, sample_weight=mask)
        self.top5_acc_metric.update_state(one_hot, preds, sample_weight=mask)
        return {
            "loss": self.loss_metric.result(),
            "top1_acc": self.top1_acc_metric.result(),
            "top5_acc": self.top5_acc_metric.result()}

    def batch_generate(self, source, start_token_id, max_gen_length):
        batch_size = tf.shape(source)[0]
        dec_input = tf.ones((batch_size, 1), dtype=tf.int32) * start_token_id
        encoder_attention_mask = tf.reduce_sum(tf.cast(source != PAD, tf.float32), axis=2) != 0
        encoder_out = self.encoder(source, mask=encoder_attention_mask, training=False)
        if self.encoder_proj is not None:
            encoder_out = self.encoder_proj(encoder_out)

        for _ in tf.range(max_gen_length-1):
            dec_output = self.decoder(
                input_ids=dec_input,
                encoder_out=encoder_out,
                encoder_attention_mask=encoder_attention_mask,
                training=False)
            logits = self.lm_head(dec_output)[:, :, :60]
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input



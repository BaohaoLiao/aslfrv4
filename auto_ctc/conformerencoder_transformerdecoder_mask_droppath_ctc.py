import tensorflow as tf
import numpy as np
from metadata import PAD, XY_POINT_LANDMARKS


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


class ConformerEncoderTransformerDecoder(tf.keras.Model):
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
        super(ConformerEncoderTransformerDecoder, self).__init__()
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
        self.ctc_head = LMHead(
            d_model=encoder_hidden_dim,
            vocab_size=vocab_size,
            activation=activation,
            name="ctc_head")

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.top1_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1_acc")
        self.top5_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")

    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]

        encoder_attention_mask = tf.reduce_sum(tf.cast(source!=PAD, tf.float32), axis=2)!=0
        encoder_out = self.encoder(source, mask=encoder_attention_mask, training=training)
        ctc_out = self.ctc_head(encoder_out)

        if self.encoder_proj is not None:
            encoder_out = self.encoder_proj(encoder_out)
        decoder_out = self.decoder(
            input_ids=target,
            encoder_out=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
            training=training
        )
        decoder_out = self.lm_head(decoder_out)
        return ctc_out, decoder_out

    @property
    def metrics(self):
        return [self.loss_metric]

    def compile(self, optimizer, loss_fn, ctc_loss_fn):
        super().compile(optimizer=optimizer)
        self.loss_fn = loss_fn
        self.ctc_loss_fn = ctc_loss_fn

    def train_step(self, batch):
        source = batch[0]
        masked_target = batch[1]
        target = batch[2]
        logits_mask = tf.reduce_sum(tf.cast(source != PAD, tf.float32), axis=2) != 0
        logits_length = tf.reduce_sum(tf.cast(logits_mask, tf.int32), axis=-1)
        target_length = tf.reduce_sum(tf.cast(target != self.pad_token_id, tf.int32), axis=-1)

        dec_input = masked_target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            ctc_out, decoder_out = self([source, dec_input], training=True)
            one_hot = tf.one_hot(dec_target, depth=self.vocab_size)
            mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_id))
            auto_loss = self.loss_fn(one_hot, decoder_out, sample_weight=mask)
            ctc_loss = self.ctc_loss_fn(
                labels=target,
                logits=ctc_out,
                label_length=target_length,
                logits_length=logits_length,
                blank_idx=self.pad_token_id,
                logits_time_major=False)
            loss = 0.8 * auto_loss + 0.2 * ctc_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_metric.update_state(loss)
        self.top1_acc_metric.update_state(one_hot, decoder_out, sample_weight=mask)
        self.top5_acc_metric.update_state(one_hot, decoder_out, sample_weight=mask)
        return {
            "loss": self.loss_metric.result(),
            "top1_acc": self.top1_acc_metric.result(),
            "top5_acc": self.top5_acc_metric.result()}

    def test_step(self, batch):
        source = batch[0]
        target = batch[2]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        _, preds = self([source, dec_input], training=False)
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

        ctc_lengths = tf.reduce_sum(tf.cast(encoder_attention_mask, tf.int32), axis=-1)
        ctc_logits = self.ctc_head(encoder_out, training=False)
        ctc_preds = tf.argmax(ctc_logits, axis=-1, output_type=tf.int32)

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
        return dec_input, ctc_preds, ctc_lengths


class TFLiteModel(tf.Module):
    def __init__(self, model, preprocess_layer, start_token_id, end_token_id, pad_token_id, max_gen_length):
        super(TFLiteModel, self).__init__()
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
        self.max_gen_length = max_gen_length
        self.model = model
        self.preprocess_layer = preprocess_layer

    @tf.function(jit_compile=True)
    def encoder(self, x):
        encoder_attention_mask = tf.reduce_sum(tf.cast(x != PAD, tf.float32), axis=2) != 0
        encoder_out = self.model.encoder(x, mask=encoder_attention_mask, training=False)
        if self.model.encoder_proj is not None:
            encoder_out = self.model.encoder_proj(encoder_out)
        return encoder_out, encoder_attention_mask

    @tf.function(jit_compile=True)
    def decoder(self, dec_input, encoder_out, encoder_attention_mask):
        dec_output = self.model.decoder(
            input_ids=dec_input,
            encoder_out=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
            training=False
        )
        return self.model.lm_head(dec_output)

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

        stop = tf.zeros((1,), dtype=tf.bool)
        for _ in tf.range(self.max_gen_length-1):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(dec_input, tf.TensorShape([1, None]))])
            logits = tf.cond(
                stop[0],
                lambda: tf.one_hot(tf.cast(dec_input, tf.int32), 60),
                lambda: self.decoder(
                    dec_input=dec_input,
                    encoder_out=encoder_out,
                    encoder_attention_mask=encoder_attention_mask)[:, :, :60])
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
            stop = tf.logical_or(stop, last_logit[0] == self.end_token_id)

        x = dec_input[0]
        idx = tf.argmax(tf.cast(tf.equal(x, self.end_token_id), tf.int32))
        idx = tf.where(tf.math.less(idx, 1), tf.constant(self.max_gen_length, dtype=tf.int64), idx)
        x = x[1:idx] # replace pad token?
        x = tf.one_hot(x, 59) # how about not in 59?
        return {'outputs': x}


class TFLiteModelEncoderOnly(tf.Module):
    def __init__(self, model, preprocess_layer, start_token_id, end_token_id, pad_token_id, max_gen_length):
        super(TFLiteModelEncoderOnly, self).__init__()
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
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

        #diff = tf.not_equal(x[:-1], x[1:])
        #adjacent_indices = tf.where(diff)[:, 0]
        #x = tf.gather(x, adjacent_indices)

        #shifted_x = tf.concat([x[1:], x[:1]], axis=0)
        shifted_x = tf.concat([x[-1:], x[:-1]], axis=0)
        is_same_as_next = tf.math.equal(x[:-1], shifted_x[:-1])
        is_same_as_next = tf.concat([is_same_as_next, [False]], axis=0)
        x = tf.boolean_mask(x, tf.math.logical_not(is_same_as_next))


        mask = x != self.pad_token_id
        x = tf.boolean_mask(x, mask, axis=0)
        mask = x != self.start_token_id
        x = tf.boolean_mask(x, mask, axis=0)
        mask = x != self.end_token_id
        x = tf.boolean_mask(x, mask, axis=0)

        x = x[:self.max_gen_length]
        x = tf.one_hot(x, 59)  # how about not in 59?
        return {'outputs': x}


class TFLiteModelEnsembleAutoCTC(tf.Module):
    def __init__(self, model, preprocess_layer, start_token_id, end_token_id, pad_token_id, max_gen_length):
        super(TFLiteModelEnsembleAutoCTC, self).__init__()
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
        self.max_gen_length = max_gen_length
        self.model = model
        self.preprocess_layer = preprocess_layer

    @tf.function(jit_compile=True)
    def encoder(self, x):
        encoder_attention_mask = tf.reduce_sum(tf.cast(x != PAD, tf.float32), axis=2) != 0
        encoder_out = self.model.encoder(x, mask=encoder_attention_mask, training=False)

        length = tf.reduce_sum(tf.cast(encoder_attention_mask, tf.int32), axis=-1)
        ctc_logits = self.model.ctc_head(encoder_out)
        ctc_pred = tf.argmax(ctc_logits, axis=-1, output_type=tf.int32)[0]
        ctc_logits = ctc_logits[0]

        diff = tf.not_equal(ctc_pred[:-1], ctc_pred[1:])
        adjacent_indices = tf.where(diff)[:, 0]
        ctc_pred = tf.gather(ctc_pred, adjacent_indices)
        ctc_logits = tf.gather(ctc_logits, adjacent_indices)

        mask = ctc_pred != self.pad_token_id
        ctc_pred = tf.boolean_mask(ctc_pred, mask, axis=0)
        ctc_logits = tf.boolean_mask(ctc_logits, mask, axis=0)
        mask = ctc_pred != self.start_token_id
        ctc_pred = tf.boolean_mask(ctc_pred, mask, axis=0)
        ctc_logits = tf.boolean_mask(ctc_logits, mask, axis=0)
        mask = ctc_pred != self.end_token_id
        ctc_pred = tf.boolean_mask(ctc_pred, mask, axis=0)
        ctc_logits = tf.boolean_mask(ctc_logits, mask, axis=0)

        if self.model.encoder_proj is not None:
            encoder_out = self.model.encoder_proj(encoder_out)
        return ctc_logits, encoder_out, encoder_attention_mask

    @tf.function(jit_compile=True)
    def decoder(self, dec_input, encoder_out, encoder_attention_mask):
        tf.print("hi")
        dec_output = self.model.decoder(
            input_ids=dec_input,
            encoder_out=encoder_out,
            encoder_attention_mask=encoder_attention_mask,
            training=False
        )
        return self.model.lm_head(dec_output)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(XY_POINT_LANDMARKS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(XY_POINT_LANDMARKS))), lambda: tf.identity(x))
        x = x[0]

        x = self.preprocess_layer(x)
        x = x[None]
        batch_size = tf.shape(x)[0]
        ctc_logits, encoder_out, encoder_attention_mask = self.encoder(x)
        ctc_logits = ctc_logits[None]
        dec_input = tf.ones((batch_size, 1), dtype=tf.int32) * self.start_token_id

        stop = tf.zeros((1,), dtype=tf.bool)
        for i in tf.range(self.max_gen_length-1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (dec_input, tf.TensorShape([1, None])),
                    (stop, tf.TensorShape([None])),
                ])
            logits = tf.cond(
                stop[0],
                lambda: tf.one_hot(tf.cast(dec_input, tf.int32), 60),
                lambda: self.decoder(
                    dec_input=dec_input,
                    encoder_out=encoder_out,
                    encoder_attention_mask=encoder_attention_mask)[:, :, :60])
            if i < tf.shape(ctc_logits)[1]:
                last_logit = logits[:, -1:, :] + ctc_logits[:, i:i+1, :60]  # TODO: prob or logit
                last_logit = tf.argmax(last_logit, axis=-1, output_type=tf.int32)
            else:
                logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
                last_logit = logits[:, -1][..., tf.newaxis]
            tf.print(i)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
            stop = tf.logical_or(stop, last_logit[0] == self.end_token_id)

        x = dec_input[0]
        idx = tf.argmax(tf.cast(tf.equal(x, self.end_token_id), tf.int32))
        idx = tf.where(tf.math.less(idx, 1), tf.constant(self.max_gen_length, dtype=tf.int64), idx)
        x = x[1:idx] # replace pad token?
        x = tf.one_hot(x, 59) # how about not in 59?
        return {'outputs': x}

import numpy as np
import tensorflow as tf
from metadata import PAD, XY_POINT_LANDMARKS

def get_initializer(initializer_range: float = 0.02) -> tf.keras.initializers.TruncatedNormal:
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)],axis=-1)
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
        self.d_model = d_model
        self.proj = tf.keras.Sequential([
            tf.keras.layers.Dense(
                d_model,
                use_bias=False,
                kernel_initializer=get_initializer(),
                activation=tf.keras.activations.gelu,
                name="emb_proj_1"
            ),
            tf.keras.layers.Dense(
                d_model,
                use_bias=False,
                kernel_initializer=get_initializer(),
                name="emb_proj_2"),
        ])
        self.learnable_position = learnable_position
        if learnable_position:
            self.position_emb = self.add_weight(
                "pos_embeddings",
                shape=[max_source_length, d_model],
                initializer=get_initializer(),
            )
        else:
            self.position_emb = positional_encoding(max_source_length, d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="emb_layernorm")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, input_frames, training):
        length = tf.shape(input_frames)[1]
        x = self.proj(input_frames)
        if self.learnable_position:
            position_ids = tf.expand_dims(tf.range(start=0, limit=length, delta=1), axis=0)
            position_embs = tf.gather(params=self.position_emb, indices=position_ids)
        else:
            position_embs = self.position_emb[tf.newaxis, :length, :]
        x = x + position_embs
        x = self.layernorm(x)
        x = self.dropout(x, training=training)
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


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim, dropout, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(mlp_dim, kernel_initializer=get_initializer(), name="dense1")
        self.dense2 = tf.keras.layers.Dense(d_model, kernel_initializer=get_initializer(), name="dense2")
        self.activation_fn = tf.keras.activations.gelu
        self.dropout = tf.keras.layers.Dropout(dropout, noise_shape=(None, 1, 1))

    def call(self, x, training):
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim, num_heads, attn_dropout, hidden_dropout, prenorm, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.prenorm = prenorm
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attn_dropout,
            name="self_attn")
        self.mlp = FeedForward(d_model, mlp_dim, hidden_dropout, name="mlp")
        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="self_attn_norm")
        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_norm")
        self.dropout = tf.keras.layers.Dropout(hidden_dropout, noise_shape=(None, 1, 1))

    def call(self, x, attention_mask, training):
        if self.prenorm:
            attn_out = self.attn_norm(x)
            attn_out = self.attention(
                query=attn_out,
                value=attn_out,
                attention_mask=attention_mask,
                training=training)
            attn_out = self.dropout(attn_out, training=training)
            attn_out = attn_out + x

            mlp_out = self.mlp_norm(attn_out)
            mlp_out = self.mlp(mlp_out, training=training)
            mlp_out = attn_out + mlp_out
        else:
            attn_out = self.attention(
                query=x,
                value=x,
                attention_mask=attention_mask,
                training=training)
            attn_out = self.dropout(attn_out, training=training)
            attn_out = self.attn_norm(attn_out + x)

            mlp_out = self.mlp(attn_out, training=training)
            mlp_out = self.mlp_norm(mlp_out + attn_out)
        return mlp_out


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, mlp_dim, num_heads, attn_dropout, hidden_dropout, prenorm, **kwargs):
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
        self.mlp = FeedForward(d_model, mlp_dim, hidden_dropout, name="mlp")

        self.self_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="self_attn_norm")
        self.cross_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="cross_attn_norm")
        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="mlp_norm")

        self.self_dropout = tf.keras.layers.Dropout(hidden_dropout, noise_shape=(None, 1, 1))
        self.cross_dropout = tf.keras.layers.Dropout(hidden_dropout, noise_shape=(None, 1, 1))

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
            mlp_out = self.mlp(mlp_out, training=training)
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

            mlp_out = self.mlp(cross_attn_out, training=training)
            mlp_out = self.mlp_norm(mlp_out + cross_attn_out)
        return mlp_out


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, max_source_length, learnable_position, num_heads, mlp_dim, emb_dropout,
        attn_dropout, hidden_dropout, prenorm, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding = LandmarkEmbedding(
            d_model=d_model,
            max_source_length=max_source_length,
            dropout=emb_dropout,
            learnable_position=learnable_position,
            name="landmark_embedding")
        self.layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                hidden_dropout=hidden_dropout,
                prenorm=prenorm,
                name=f"encoder_layers.{i}") for i in range(num_layers)]
        self.final_norm = None
        if prenorm:
            self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")

    def call(self, input_frames, training):
        attention_mask = tf.reduce_sum(tf.cast(input_frames!=PAD, tf.float32), axis=2)!=0  # B x T
        extended_attention_mask = _expand_mask(attention_mask)

        x = self.embedding(input_frames, training=training)
        for _, encoder_layer in enumerate(self.layers):
            x = encoder_layer(
                x,
                attention_mask=extended_attention_mask,
                training=training)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x, attention_mask


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers, d_model, max_target_length, learnable_position, num_heads, mlp_dim,
        emb_dropout, attn_dropout, hidden_dropout, pad_token_id, prenorm, **kwargs):
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


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_encoder_layers: int,
        encoder_hidden_dim: int,
        encoder_mlp_dim: int,
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
        prenorm: bool):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=encoder_hidden_dim,
            max_source_length=max_source_length,
            learnable_position=learnable_position,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            emb_dropout=emb_dropout,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            prenorm=prenorm,
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
            name="decoder")

        self.encoder_proj = None
        if encoder_hidden_dim != decoder_hidden_dim:
            self.encoder_proj = tf.keras.layers.Dense(
                decoder_hidden_dim, kernel_initializer=get_initializer(), use_bias=False, name="encoder_proj")
        lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
        self.lm_head = tf.keras.layers.Dense(
            vocab_size, use_bias=False, kernel_initializer=lm_head_initializer, name='lm_head')

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.top1_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1_acc")
        self.top5_acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")

    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]

        encoder_out, encoder_attention_mask = self.encoder(source, training=training)
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
        target = batch[1]
        dec_input = target[:, :-1]
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
        encoder_out, encoder_attention_mask = self.encoder(source, training=False)
        if self.encoder_proj is not None:
            encoder_out = self.encoder_proj(encoder_out)

        for _ in tf.range(max_gen_length-1):
            dec_output = self.decoder(
                input_ids=dec_input,
                encoder_out=encoder_out,
                encoder_attention_mask=encoder_attention_mask,
                training=False)
            logits = self.lm_head(dec_output)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

    def efficient_generate(self, source, start_token_id, end_token_id, max_gen_length):
        batch_size = tf.shape(source)[0]
        dec_input = tf.ones((batch_size, 1), dtype=tf.int32) * start_token_id
        encoder_out, encoder_attention_mask = self.encoder(source, training=False)
        if self.encoder_proj is not None:
            encoder_out = self.encoder_proj(encoder_out)

        for _ in tf.range(max_gen_length - 1):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(dec_input, tf.TensorShape([1, None]))])
            dec_output = self.decoder(
                input_ids=dec_input,
                encoder_out=encoder_out,
                encoder_attention_mask=encoder_attention_mask,
                training=False)
            logits = self.lm_head(dec_output)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
            if (last_logit == end_token_id):
                break
        return dec_input
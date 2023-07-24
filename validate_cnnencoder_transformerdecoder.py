import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from Levenshtein import distance as Lev_distance

from models.cnnencoder_transformerdecoder import CNNEncoderTransformerDecoder, TFLiteModel
from metadata import XY_POINT_LANDMARKS
from data import filter_nans_tf, tf_nan_mean, tf_nan_std, decode_fn
from train_cnnencoder_transformerdecoder import parse_args

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, max_source_length, **kwargs):
        super().__init__(**kwargs)
        self.max_source_length = max_source_length

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, len(XY_POINT_LANDMARKS)], dtype=tf.float32),), )
    def call(self, inputs):
        n_frames = tf.shape(inputs)[0]
        frame_dim = tf.shape(inputs)[1]
        x = tf.transpose(tf.reshape(inputs, [n_frames, 2, frame_dim // 2]), perm=[0, 2, 1])  # B x C//2 x 2
        x = filter_nans_tf(x)
        x = x[None, ...]

        mean = tf_nan_mean(tf.gather(x, [0], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std

        if n_frames > self.max_source_length:
            #x = x[:, :self.max_source_length]
            x = tf.image.resize(
                x,
                [self.max_source_length, frame_dim//2],
                method=tf.image.ResizeMethod.BILINEAR,
            )
        if n_frames < 3:
            x = tf.pad(x, [[0, 0], [0, 3-n_frames], [0, 0], [0, 0]])

        length = tf.shape(x)[1]
        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))

        dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
                      lambda: tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1, length, frame_dim)),
            tf.reshape(dx, (-1, length, frame_dim)),
            tf.reshape(dx2, (-1, length, frame_dim)),
        ], axis=-1)
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x[0]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(args)

    model = CNNEncoderTransformerDecoder(
        num_encoder_layers=args.num_encoder_layers,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_mlp_dim=args.encoder_mlp_dim,
        encoder_num_heads=args.encoder_num_heads,
        encoder_conv_dim=args.encoder_conv_dim,
        encoder_kernel_size=args.encoder_kernel_size,
        encoder_dilation_rate=args.encoder_dilation_rate,
        max_source_length=args.max_source_length,
        num_decoder_layers=args.num_decoder_layers,
        vocab_size=args.vocab_size,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_mlp_dim=args.decoder_mlp_dim,
        decoder_num_heads=args.decoder_num_heads,
        max_target_length=args.max_target_length,
        pad_token_id=args.pad_token_id,
        emb_dropout=args.emb_dropout,
        attn_dropout=args.attn_dropout,
        hidden_dropout=args.hidden_dropout,
        learnable_position=args.learnable_position,
        prenorm=args.prenorm,
        activation=args.activation,
    )
    preprocess_layer = PreprocessLayer(args.max_source_length)
    tflitemodel = TFLiteModel(
        model, preprocess_layer, args.start_token_id, args.end_token_id, args.pad_token_id, args.max_gen_length)


    with open(os.path.join(args.data_dir, "character_to_prediction_index.json"), "r") as f:
        char_to_num = json.load(f)
    char_to_num[args.pad_token] = args.pad_token_id
    char_to_num[args.start_token] = args.start_token_id
    char_to_num[args.end_token] = args.end_token_id
    num_to_char = {j: i for i, j in char_to_num.items()}

    logging.info("--------- Corner Case --------------")
    frames = [
        tf.zeros(shape=[0, len(XY_POINT_LANDMARKS)]),  # length=0
        tf.random.uniform(shape=[1, len(XY_POINT_LANDMARKS)]),
        tf.zeros([1, len(XY_POINT_LANDMARKS)]),  # all zeros
        tf.fill((1, len(XY_POINT_LANDMARKS)), tf.constant(np.nan)),  # all nans
        tf.random.uniform(shape=[600, len(XY_POINT_LANDMARKS)])  # larger than max source length
    ]
    logging_infos = [
        "length=0",
        "length=1",
        "all zeros",
        "all nans",
        "larger than max_source_length",
    ]
    for i, frame in enumerate(frames):
        output = tflitemodel(inputs=frame)
        prediction_str = "".join([num_to_char.get(s, "") for s in np.argmax(output["outputs"], axis=1)])
        logging.info(f"{logging_infos[i]}\tShape: {tf.shape(frame)}\tPrediction: {prediction_str}")
    logging.info("--------- Success --------------")

    fp16_keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel)
    fp16_keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    fp16_keras_model_converter.target_spec.supported_types = [tf.float16]
    fp16_keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    fp16_tflite_model = fp16_keras_model_converter.convert()
    with open(os.path.join(args.output_dir, 'model.tflite'), 'wb') as f:
        f.write(fp16_tflite_model)

if __name__ == "__main__":
    main()

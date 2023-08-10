import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from Levenshtein import distance as Lev_distance

from auto_ctc.conformerencoder_transformerdecoder_mask_droppath_ctc import (
    ConformerEncoderTransformerDecoder,
    TFLiteModel,
)

from metadata import XY_POINT_LANDMARKS, PAD
from data import filter_nans_tf, tf_nan_mean, tf_nan_std, decode_fn
from train_cnnencoder_transformerdecoder_mask_ctc_distributed import parse_args

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

        mean = tf_nan_mean(tf.gather(x, [6], axis=2), axis=[1, 2], keepdims=True)
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
        #x = tf.pad(x, [[0, 0], [0, self.max_source_length - length], [0, 0]], constant_values=PAD)
        return x[0]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ignore_keys = ["checkpoint_path", "output_dir", "max_gen_length", "data_dir"]
    with open(os.path.join(os.path.dirname(args.checkpoint_path), 'args.json'), 'r') as f:
        trained_args = json.load(f)
    for key, value in trained_args.items():
        if (key not in ignore_keys):
            setattr(args, key, value)
    with open(os.path.join(args.output_dir, "valid_args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)

    if args.model_arch == "conformerencoder_transformerdecoder_ctc":
        model = ConformerEncoderTransformerDecoder(
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
            activation=args.activation,)
    virtual_intput = (
        np.zeros((1, args.max_source_length, 3 * len(XY_POINT_LANDMARKS)), dtype=np.float32),
        np.zeros((1, args.max_target_length), dtype=np.int32)
    )
    model(virtual_intput)
    #logging.info(f"{tf.shape(model(virtual_intput))}")
    model.load_weights(args.checkpoint_path)
    model.save(os.path.join(args.output_dir, 'model.keras'))

    preprocess_layer = PreprocessLayer(args.max_source_length)
    tflitemodel = TFLiteModel(
        model, preprocess_layer, args.start_token_id, args.end_token_id, args.pad_token_id, args.max_gen_length)


    with open(os.path.join(args.data_dir, "character_to_prediction_index.json"), "r") as f:
        char_to_num = json.load(f)
    char_to_num[args.pad_token] = args.pad_token_id
    char_to_num[args.start_token] = args.start_token_id
    char_to_num[args.end_token] = args.end_token_id
    num_to_char = {j: i for i, j in char_to_num.items()}

    if args.fold == "all":
        args.fold = "0"
    valid_tffiles = [os.path.join(args.data_dir, f"fold{args.fold}.tfrecord")]
    logging.info(f"Valid files: {valid_tffiles}")
    ds = tf.data.TFRecordDataset(valid_tffiles, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(decode_fn, tf.data.AUTOTUNE)
    ds = ds.padded_batch(1).prefetch(tf.data.AUTOTUNE)
    batches = [batch for batch in ds]
    logging.info(f"#VALIDATION SAMPLES: {len(batches)}")

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

    global_n, global_d = 0, 0
    local_score = 0
    start_time = datetime.now()
    n_samples = 10
    for i, batch in enumerate(batches[:n_samples]):
        output = tflitemodel(inputs=batch[0][0])
        prediction_str = "".join([num_to_char.get(s, "") for s in np.argmax(output["outputs"], axis=1)])
        target = batch[1][0].numpy().decode('utf-8')
        lv_distance = Lev_distance(target, prediction_str)
        global_n += len(target)
        global_d += lv_distance
        local_score += np.clip((len(target) - lv_distance) / len(target), a_min=0, a_max=1)

        logging.info(f"{i + 1}/{len(batches)}\tTARGET: {target}\tPREDICTION: {prediction_str}\tLV: {lv_distance}\t"
                     f"LENGTH: {len(target)}/{len(batch[0][0])}")
    score = np.clip((global_n - global_d) / global_n, a_min=0, a_max=1)
    logging.info(f"GLOBAL SCORE: {score}, GLOBAL_LENGTH: {global_n}, GLOBAL_DISTANCE: {global_d}, "
                 f"LOCAL SCORE: {local_score / n_samples}")

    end_time = datetime.now()
    logging.info('Duration: {}'.format(end_time - start_time))

    fp16_keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel)
    fp16_keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    fp16_keras_model_converter.target_spec.supported_types = [tf.float16]
    fp16_keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    fp16_tflite_model = fp16_keras_model_converter.convert()
    with open(os.path.join(args.output_dir, 'model.tflite'), 'wb') as f:
        f.write(fp16_tflite_model)

if __name__ == "__main__":
    main()

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from metadata import XY_POINT_LANDMARKS
from train_cnnencoder_transformerdecoder import parse_args
from models.cnnencoder_transformerdecoder import CNNEncoderTransformerDecoder
from mask.conformerencoder_transformerdecoder_mask import ConformerEncoderTransformerDecoder


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ignore_keys = ["checkpoint_path", "output_dir", "data_dir", "max_gen_length"]
    paths = args.checkpoint_path.split(":")

    with open(os.path.join(os.path.dirname(paths[0]), 'args.json'), 'r') as f:
        trained_args = json.load(f)
    for key, value in trained_args.items():
        if (key not in ignore_keys):
            setattr(args, key, value)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)

    if args.model_arch == "cnnencoder_transformerdecoder":
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
            activation=args.activation)
    elif args.model_arch == "conformerencoder_transformerdecoder":
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
            activation=args.activation)
    virtual_intput = (
        np.zeros((1, args.max_source_length, 3 * len(XY_POINT_LANDMARKS)), dtype=np.float32),
        np.zeros((1, args.max_target_length), dtype=np.int32)
    )
    logging.info(f"{tf.shape(model(virtual_intput))}")

    assert len(paths) > 1
    def load_weights(model, path):
        model.load_weights(path)
        return model.get_weights()

    weights = [load_weights(model, path) for path in paths]
    averaged_weights = np.mean(weights, axis=0)
    model.set_weights(averaged_weights)
    model.save_weights(os.path.join(args.output_dir, "average.h5"))


if __name__ == "__main__":
    main()


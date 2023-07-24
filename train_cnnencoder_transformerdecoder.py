import os
import sys
import json
import random
import argparse
import logging
import numpy as np
import tensorflow as tf

from data import load_dataset
#from optimizer import LRInverseSqrtScheduler
#from display import DisplayOutputs


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train ASLFR models")
    # Output args
    parser.add_argument("--output_dir", type=str, required=True)
    # Data args
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fold", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=10)
    # Model args
    parser.add_argument("--model_arch",
                        choices=["transformer", "cnnencoder_transformerdecoder"],
                        default="transformer")
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--encoder_hidden_dim", type=int, default=384)
    parser.add_argument("--encoder_mlp_dim", type=int, default=768)
    parser.add_argument("--encoder_num_heads", type=int, default=6)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=62)
    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_num_heads", type=int, default=4)
    parser.add_argument("--decoder_mlp_dim", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=34)
    parser.add_argument("--emb_dropout", type=float, default=0.)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dropout", type=float, default=0.2)
    parser.add_argument("--learnable_position", action="store_true")
    parser.add_argument("--prenorm", action="store_true")
    parser.add_argument("--pad_token_id", type=int, default=59)
    parser.add_argument("--start_token_id", type=int, default=60)
    parser.add_argument("--end_token_id", type=int, default=61)
    parser.add_argument("--pad_token", type=str, default="P")
    parser.add_argument("--start_token", type=str, default="S")
    parser.add_argument("--end_token", type=str, default="E")
    parser.add_argument("--activation", type=str, default="swish")
    parser.add_argument("--encoder_conv_dim", type=int, default=768)
    parser.add_argument("--encoder_kernel_size", type=int, default=17)
    parser.add_argument("--encoder_dilation_rate", type=int, default=1)
    # Training args
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_supplement_data", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_norm", type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=int, default=0.1)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    # Data augmentation
    parser.add_argument("--flip", type=float, default=0.)
    parser.add_argument("--resample", type=float, default=0.)
    parser.add_argument("--affine", type=float, default=0.)
    parser.add_argument("--temporal_mask", type=float, default=0.)
    parser.add_argument("--spatial_mask", type=float, default=0.)
    parser.add_argument("--use_speed", action="store_true")
    parser.add_argument("--use_acceleration", action="store_true")
    # For validation
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--max_gen_length", type=int, default=34)
    args = parser.parse_args()
    return args
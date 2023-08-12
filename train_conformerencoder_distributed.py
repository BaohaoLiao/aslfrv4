import os
import sys
import json
import random
import argparse
import logging
import numpy as np
import tensorflow as tf

from optimizer import LRInverseSqrtScheduler
from encoderonly.data_encoderonly import load_dataset
from encoderonly.displayv2 import DisplayOutputs
from ctc.conformerencoder import Conformer
from metadata import XY_POINT_LANDMARKS

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
                        choices=["conformerencoder"],
                        default="conformerencoder")
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--encoder_hidden_dim", type=int, default=384)
    parser.add_argument("--encoder_mlp_dim", type=int, default=768)
    parser.add_argument("--encoder_num_heads", type=int, default=6)
    parser.add_argument("--encoder_conv_dim", type=int, default=768)
    parser.add_argument("--encoder_kernel_size", type=int, default=17)
    parser.add_argument("--encoder_dilation_rate", type=int, default=1)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=31)
    parser.add_argument("--vocab_size", type=int, default=60)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dropout", type=float, default=0.2)
    parser.add_argument("--pad_token_id", type=int, default=59)
    parser.add_argument("--pad_token", type=str, default="P")
    parser.add_argument("--activation", type=str, default="swish")
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
    parser.add_argument("--concat", type=float, default=0.)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_valid", type=int, default=6432)
    # For validation
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--max_gen_length", type=int, default=34)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)

    devices = tf.config.experimental.list_physical_devices('GPU')
    logging.info(f"GPUs available: {len(devices)}")
    strategy = tf.distribute.MirroredStrategy()
    logging.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    def seed_everything(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    seed_everything(seed=args.seed)

    with open(os.path.join(args.data_dir, "character_to_prediction_index.json"), "r") as f:
        char_to_num = json.load(f)
    char_to_num[args.pad_token] = args.pad_token_id
    num_to_char = {j: i for i, j in char_to_num.items()}

    train_tffiles = []
    valid_tffiles = []
    if args.fold == "all":
        for i in range(args.num_folds):
            train_tffiles.append(os.path.join(args.data_dir, f"fold{i}.tfrecord"))
        valid_tffiles.append(train_tffiles[0])
    else:
        for i in range(args.num_folds):
            if str(i) == args.fold:
                valid_tffiles.append(os.path.join(args.data_dir, f"fold{i}.tfrecord"))
            else:
                train_tffiles.append(os.path.join(args.data_dir, f"fold{i}.tfrecord"))
    logging.info(f"Train files: {train_tffiles}")
    logging.info(f"Valid files: {valid_tffiles}")

    train_dataset = load_dataset(
        tfrecords=train_tffiles,
        args=args,
        char_to_num=char_to_num,
        augment=True,
        repeat=True,
        shuffle=30000,
        drop_remainder=True
    )
    val_dataset = load_dataset(
        tfrecords=valid_tffiles,
        args=args,
        char_to_num=char_to_num,
        augment=False,
        repeat=False,
        shuffle=False,
        drop_remainder=False
    )

    # TODO: automatically count
    if args.fold != "all":
        num_train = 64329 - args.num_valid
    else:
        num_train = 64329
        if args.num_folds == 20:
            num_train += 50927
    num_valid = args.num_valid
    steps_per_epoch = num_train // args.batch_size
    total_steps = num_train * args.num_epochs // args.batch_size

    with strategy.scope():
        if args.model_arch == "conformerencoder":
            model = Conformer(
                num_encoder_layers=args.num_encoder_layers,
                encoder_hidden_dim=args.encoder_hidden_dim,
                encoder_mlp_dim=args.encoder_mlp_dim,
                encoder_num_heads=args.encoder_num_heads,
                encoder_conv_dim=args.encoder_conv_dim,
                encoder_kernel_size=args.encoder_kernel_size,
                encoder_dilation_rate=args.encoder_dilation_rate,
                max_source_length=args.max_source_length,
                vocab_size=args.vocab_size,
                pad_token_id=args.pad_token_id,
                attn_dropout=args.attn_dropout,
                hidden_dropout=args.hidden_dropout,
                activation=args.activation)

        virtual_intput = np.zeros((1, args.max_source_length, 3 * len(XY_POINT_LANDMARKS)), dtype=np.float32),
        logging.info(f"{tf.shape(model(virtual_intput))}")
        logging.info(model.summary())

        learning_rate = LRInverseSqrtScheduler(args.lr, warmup_steps=int(args.warmup_ratio * total_steps))
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
            clipnorm=args.max_norm,
            weight_decay=args.weight_decay,
        )

        def CTCLoss(labels, logits, label_length, logits_length, blank_idx, logits_time_major):
            loss = tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=logits_length,
                blank_index=blank_idx,
                logits_time_major=logits_time_major
            )
            loss = tf.reduce_mean(loss)
            return loss

        model.compile(optimizer=optimizer, loss_fn=CTCLoss)
        if args.resume is not None:
            logging.info(f"Resume from {args.resume}")
            model.load_weights(args.resume)
            if train_dataset is not None:
                model.evaluate(train_dataset.take(steps_per_epoch))
            if val_dataset is not None:
                model.evaluate(val_dataset)

    display_callback = DisplayOutputs(
        model,
        val_dataset,
        num_to_char,
        pad_token=args.pad_token,
        pad_token_id=args.pad_token_id)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, "checkpoint_epoch{epoch:03d}.h5"),
        save_weights_only=True,
        save_freq=steps_per_epoch,
    )
    callbacks = [display_callback, model_checkpoint_callback]

    _ = model.fit(
        train_dataset,
        verbose=args.verbose,
        validation_data=val_dataset,
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=-(num_valid // -args.batch_size),
        callbacks=callbacks)

if __name__ == "__main__":
    main()
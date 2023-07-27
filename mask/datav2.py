import tensorflow as tf
import numpy as np
from metadata import (
    XY_POINT_LANDMARKS,
    LHAND_IDXS,
    RHAND_IDXS,
    LLIP_IDXS,
    RLIP_IDXS,
    LEYE_IDXS,
    REYE_IDXS,
    LNOSE_IDXS,
    RNOSE_IDXS,
    PAD,
)


def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in XY_POINT_LANDMARKS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks = ([tf.sparse.to_dense(features[COL]) for COL in XY_POINT_LANDMARKS])
    landmarks = tf.transpose(landmarks) # T x C
    return landmarks, phrase

def encode(landmarks, phrase, table, max_target_length, start_token="S", end_token="E", pad_token_id=59):
    phrase = start_token + phrase + end_token
    phrase = tf.strings.bytes_split(phrase)
    phrase = table.lookup(phrase)
    #phrase = tf.pad(
    #    phrase, paddings=[[0, max_target_length - tf.shape(phrase)[0]]], mode = 'CONSTANT',
    #    constant_values = pad_token_id)
    return landmarks, phrase

def apply_mask(landmarks, phrase, mask_prob, mask_token_id, random_token_prob):
    if mask_prob > 0.:
        masked_char_ids = tf.identity(phrase)
        mask_range_start = tf.constant(1, dtype=tf.int32)
        mask_range_end = tf.shape(masked_char_ids)[0] - 1
        mask_range = tf.range(mask_range_start, mask_range_end)
        mask = tf.random.uniform(shape=tf.shape(masked_char_ids[mask_range]), dtype=tf.float32) < mask_prob
        mask_indices = tf.boolean_mask(mask_range, mask)
        mask_values = tf.fill(tf.shape(mask_indices), mask_token_id)
        masked_char_ids = tf.tensor_scatter_nd_update(masked_char_ids, tf.expand_dims(mask_indices, axis=1),
                                                      mask_values)
        random_replace = mask & (tf.random.uniform(shape=tf.shape(masked_char_ids[mask_range]),
                                                   dtype=tf.float32) < random_token_prob)
        random_char_ids = tf.random.uniform(shape=tf.shape(masked_char_ids[mask_range]), minval=0,
                                            maxval=59, dtype=tf.int32)
        masked_char_ids = tf.tensor_scatter_nd_update(
            masked_char_ids,
            tf.expand_dims(tf.boolean_mask(mask_range, random_replace), axis=1),
            random_char_ids)
        return tf.identity(landmarks), masked_char_ids, tf.identity(phrase)
    else:
        return tf.identity(landmarks), tf.identity(phrase), tf.identity(phrase)

def filter_nans_tf(x):
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(x), axis=[-2,-1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x

def interp1d_(x, target_len, method):
    target_len = tf.maximum(1, target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]),'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'bicubic')
            else:
                x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "bilinear")
    return x

def resample(x, rate=(0.8, 1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate*tf.cast(length,tf.float32), tf.int32)
    new_x = interp1d_(x, new_size, method="random")
    return new_x

def flip_lr(x):
    x,y = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x,y], -1)
    new_x = tf.transpose(new_x, [1,0,2])
    lhand = tf.gather(new_x, LHAND_IDXS, axis=0)
    rhand = tf.gather(new_x, RHAND_IDXS, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND_IDXS)[...,None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND_IDXS)[...,None], lhand)
    llip = tf.gather(new_x, LLIP_IDXS, axis=0)
    rlip = tf.gather(new_x, RLIP_IDXS, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP_IDXS)[...,None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP_IDXS)[...,None], llip)
    leye = tf.gather(new_x, LEYE_IDXS, axis=0)
    reye = tf.gather(new_x, REYE_IDXS, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LEYE_IDXS)[...,None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(REYE_IDXS)[...,None], leye)
    lnose = tf.gather(new_x, LNOSE_IDXS, axis=0)
    rnose = tf.gather(new_x, RNOSE_IDXS, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LNOSE_IDXS)[...,None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RNOSE_IDXS)[...,None], lnose)
    new_x = tf.transpose(new_x, [1,0,2])
    return new_x

def spatial_random_affine(
    xy,
    scale  = (0.8,1.2),
    shear = (-0.15,0.15),
    shift  = (-0.1,0.1),
    degree = (-30,30),
):
    center = tf.constant([0.5,0.5])
    if scale is not None:
        scale = tf.random.uniform((), *scale)
        xy = scale * xy
    if shear is not None:
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = tf.identity([[1., shear_x], [shear_y, 1.]])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
    if degree is not None:
        xy -= center
        degree = tf.random.uniform((), *degree)
        radian = degree/180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([[c,s], [-s, c],])
        xy = xy @ rotate_mat
        xy = xy + center
    if shift is not None:
        shift = tf.random.uniform((), *shift)
        xy = xy + shift
    return xy

def temporal_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    l = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l-mask_size,1,l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(
        x,
        tf.range(mask_offset, mask_offset+mask_size)[...,None],
        tf.fill([mask_size, len(XY_POINT_LANDMARKS)//2, 2],
        mask_value)
    )
    return x

def spatial_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x<x[...,0]) & (x[...,0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y<x[...,1]) & (x[...,1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[...,None], mask_value, x)
    return x

def augment_fn(x, args, max_source_length):
    if tf.random.uniform(()) < args.resample:
        x = resample(x, (0.5,1.5))
    if tf.random.uniform(()) < args.flip:
        x = flip_lr(x)
    if tf.shape(x)[0] > max_source_length:
        x = tf.image.resize(x, (max_source_length, tf.shape(x)[1]), 'bilinear')
    if tf.random.uniform(()) < args.affine:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < args.temporal_mask:
        x = temporal_mask(x)
    if tf.random.uniform(()) < args.spatial_mask:
        x = spatial_mask(x)
    return x

def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) \
           / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_source_length, use_speed, use_acceleration, **kwargs):
        super().__init__(**kwargs)
        self.max_source_length = max_source_length
        self.use_speed = use_speed
        self.use_acceleration = use_acceleration

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs

        #mean = tf_nan_mean(tf.gather(x, [0], axis=2), axis=[1, 2], keepdims=True)
        mean = tf_nan_mean(tf.gather(x, [6], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std

        n_frames = tf.shape(x)[1]
        if n_frames > self.max_source_length:
            #x = tf.image.resize(x, (self.max_source_length, len(XY_POINT_LANDMARKS)//2), 'bilinear')
            x = x[:, :self.max_source_length]
        if n_frames < 3:
            x = tf.pad(x, [[0, 0], [0, 3-n_frames], [0, 0], [0, 0]])

        length = tf.shape(x)[1]
        output = tf.reshape(x, (-1, length, len(XY_POINT_LANDMARKS)))
        if self.use_speed:
            dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))
            output = tf.concat([
                output,
                tf.reshape(dx, (-1, length, len(XY_POINT_LANDMARKS))),
            ], axis=-1)
        if self.use_acceleration:
            dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
                      lambda: tf.zeros_like(x))
            output = tf.concat([
                output,
                tf.reshape(dx2, (-1, length, len(XY_POINT_LANDMARKS))),
            ], axis=-1)

        output = tf.where(tf.math.is_nan(output), tf.constant(0., output.dtype), output)
        return output


def preprocess(x, y, z, args, augment, max_source_length):
    x = tf.ensure_shape(x, (None, len(XY_POINT_LANDMARKS))) # B x C
    n_frames = tf.shape(x)[0]
    frame_dim = tf.shape(x)[1]
    x = tf.transpose(tf.reshape(x, [n_frames, 2, frame_dim // 2]), perm=[0, 2, 1])  # B x C//2 x 2

    x = filter_nans_tf(x)
    if augment:
        x = augment_fn(x, args=args, max_source_length=max_source_length)
    x = tf.ensure_shape(x, (None, len(XY_POINT_LANDMARKS) // 2, 2))
    x = Preprocess(
            max_source_length=max_source_length,
            use_speed=args.use_speed,
            use_acceleration=args.use_acceleration
        )(x)[0]
    return tf.cast(x, tf.float32), y, z

def maybe_concat_samples(sample1, sample2, concat_prob):
    frame1, phrase1, label1 = sample1
    frame2, phrase2, label2 = sample2
    if tf.random.uniform(()) < concat_prob:
        frame = tf.concat([frame1, frame2], axis=0)
        phrase = tf.concat([phrase1[:-1], [0], phrase2[1:]], axis=0)  # delete <s>, </s> from phrase1
        label = tf.concat([label1[:-1], [0], label2[1:]], axis=0)  # delete <s>, </s> from phrase1
    else:
        frame, phrase, label = tf.identity(frame1), tf.identity(phrase1), tf.identity(label1)
    return frame, phrase, label


def load_dataset(
    tfrecords,
    args,
    char_to_num,
    augment,
    repeat=False,
    shuffle=False,
    drop_remainder=False,
):
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=list(char_to_num.keys()),
            values=list(char_to_num.values()),
        ),
        default_value=tf.constant(-1),
        name="class_weight"
    )

    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(decode_fn, tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: encode(x, y, table, args.max_target_length), tf.data.AUTOTUNE)

    if augment:
        mask_prob = args.mask_prob
    else:
        mask_prob = 0.
    ds = ds.map(lambda x, y: apply_mask(
        x, y, mask_prob=mask_prob, mask_token_id=args.mask_token_id, random_token_prob=args.random_token_prob),
        tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle, reshuffle_each_iteration=True)
        options = tf.data.Options()
        options.experimental_deterministic = (False)
        ds = ds.with_options(options)
        if args.concat > 0.:
            ds_pairs = tf.data.Dataset.zip((ds, ds.skip(1)))
            ds = ds_pairs.map(lambda xyz1, xyz2: maybe_concat_samples(xyz1, xyz2, args.concat))

    ds = ds.map(
        lambda x, y, z: preprocess(x, y, z, args=args, augment=augment, max_source_length=args.max_source_length),
        tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    if args.use_speed and args.use_acceleration:
        x_padded_shape = [args.max_source_length, 3 * len(XY_POINT_LANDMARKS)]
    elif args.use_speed or args.use_acceleration:
        x_padded_shape = [args.max_source_length, 2 * len(XY_POINT_LANDMARKS)]
    else:
        x_padded_shape = [args.max_source_length, len(XY_POINT_LANDMARKS)]

    ds = ds.padded_batch(
        args.batch_size,
        padding_values=(PAD, args.pad_token_id, args.pad_token_id),
        padded_shapes=(x_padded_shape, [args.max_target_length], [args.max_target_length]),  # TODO: check encoder
        drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
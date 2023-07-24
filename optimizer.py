import tensorflow as tf

class LRInverseSqrtScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, min_lr=1e-7):
        super().__init__()
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.lr_step = tf.cast((max_lr - min_lr) / warmup_steps, tf.float32)
        self.decay_factor = tf.cast(max_lr * warmup_steps ** 0.5, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.min_lr + step * self.lr_step,
            lambda: self.decay_factor * step ** -0.5
        )
import logging
import tensorflow as tf
from Levenshtein import distance as Lev_distance

class DisplayOutputs(tf.keras.callbacks.Callback):
    def __init__(self, model, val_dataset, idx_to_token, pad_token, pad_token_id):
        self.model = model
        self.batches = [batch for batch in val_dataset]
        self.idx_to_char = idx_to_token
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f' learning rate: {self.model.optimizer.learning_rate.numpy():.5e}')
        lv_distances = []
        for batch in self.batches:
            source = batch[0]
            target = batch[1].numpy()
            batch_size = tf.shape(source)[0]
            preds = self.model(source, training=False)
            for i in range(batch_size):
                target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
                pred_ids = self.decode(preds[i])
                prediction = ""
                for idx in pred_ids:
                    prediction += self.idx_to_char[idx]
                target_text = target_text.replace(self.pad_token, '')
                prediction = prediction.replace(self.pad_token, '')
                lv_distances.append({
                    "target": target_text,
                    "prediction": prediction,
                    "lv": Lev_distance(target_text, prediction)
                })
        score = 0
        global_length, global_dist = 0, 0
        for i, lv in enumerate(lv_distances):
            score += (len(lv["target"]) - lv["lv"]) / len(lv["target"])
            global_length += len(lv["target"])
            global_dist += lv["lv"]
            if i <= 10:
                logging.info(lv)
        score /= len(lv_distances)
        global_score = (global_length - global_dist) / global_length
        logging.info(f"Epoch{epoch + 1}: validation score, local: {score}, global: {global_score}")

    def decode(self, pred):
        x = tf.argmax(pred, axis=1)
        diff = tf.not_equal(x[:-1], x[1:])
        adjacent_indices = tf.where(diff)[:, 0]
        x = tf.gather(x, adjacent_indices)
        mask = x != self.pad_token_id
        x = tf.boolean_mask(x, mask, axis=0)
        return x
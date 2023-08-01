import logging
import tensorflow as tf
from Levenshtein import distance as Lev_distance

class DisplayOutputs(tf.keras.callbacks.Callback):
    def __init__(
            self, model, val_dataset, idx_to_token, pad_token_id, start_token_id, end_token_id,  pad_token, start_token, end_token,
            max_target_length):
        self.model = model
        self.batches = [batch for batch in val_dataset]
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.idx_to_char = idx_to_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.max_target_length = max_target_length

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f'Learning rate: {self.model.optimizer.learning_rate.numpy():.5e}')
        if epoch > 0:
            lv_distances = []
            for batch in self.batches:
                source = batch[0]
                target = batch[1].numpy()
                batch_size = tf.shape(source)[0]
                preds, ctc_preds, ctc_lengths = self.model.batch_generate(source, self.start_token_id, self.max_target_length)

                preds = preds.numpy()
                for i in range(batch_size):
                    target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
                    prediction = ""
                    for idx in preds[i, :]:
                        prediction += self.idx_to_char[idx]
                        if idx == self.end_token_id:
                            break
                    target_text = target_text.replace(self.pad_token, '').\
                                      replace(self.start_token, '').replace(self.end_token, '')
                    prediction = prediction.replace(self.pad_token, ''). \
                                      replace(self.start_token, '').replace(self.end_token, '')

                    ctc_pred_ids = self.decode(ctc_preds[i][:ctc_lengths[i]])
                    ctc_pred_ids = ctc_pred_ids.numpy()
                    ctc_prediction = ""
                    for idx in ctc_pred_ids:
                        ctc_prediction += self.idx_to_char[idx]
                    ctc_prediction = ctc_prediction.replace(self.pad_token, ''). \
                        replace(self.start_token, '').replace(self.end_token, '')

                    lv_distances.append({
                        "target": target_text,
                        "prediction": prediction,
                        "ctc_prediction": ctc_prediction,
                        "lv": Lev_distance(target_text, prediction),
                        "ctc_lv": Lev_distance(target_text, ctc_prediction)
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

            ctc_score = 0
            ctc_global_length, ctc_global_dist = 0, 0
            for i, lv in enumerate(lv_distances):
                ctc_score += (len(lv["target"]) - lv["ctc_lv"]) / len(lv["target"])
                ctc_global_length += len(lv["target"])
                ctc_global_dist += lv["ctc_lv"]
            ctc_score /= len(lv_distances)
            ctc_global_score = (ctc_global_length - ctc_global_dist) / ctc_global_length

            logging.info(f"Epoch{epoch + 1}: auto score, local: {score}, ctc score: {ctc_global_score}")

    def decode(self, x):
        diff = tf.not_equal(x[:-1], x[1:])
        adjacent_indices = tf.where(diff)[:, 0]
        x = tf.gather(x, adjacent_indices)
        mask = x != self.pad_token_id
        x = tf.boolean_mask(x, mask, axis=0)
        return x
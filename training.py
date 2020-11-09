from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from random import randrange
from sklearn.metrics import make_scorer
# https://keras.io/api/models/model/

import time
import tensorflow as tf
import numpy as np
import os


class train_classifier:

    def __init__(self,
                 x_train, y_train,
                 x_valid, y_valid,
                 n_batch=64,
                 val_size=0.1):
        """
        str file_path: path to save the object
        """
        self.val_size = val_size
        self.n_batch = n_batch

        self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_ds = self.train_ds.shuffle(buffer_size=1024).batch(self.n_batch)

        self.val_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        self.val_ds = self.val_ds.batch(self.n_batch)

    def get_batch(self, i, array):

        start_id = i * self.n_batch
        end_id = min((i + 1) * self.n_batch, array.shape[0])

        batch_x = array[start_id:end_id]

        return batch_x

    def train_classifier(self,
                         model,
                         optimizer=tf.keras.optimizers.Adam(learning_rate=10e-3),
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False)):  # do f-measure

        start_time = time.time()

        tot_loss = 0
        iterations = 0
        cumulPrd = None
        cumulY = None

        for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train,
                               training=True)
                # loss between original variable and prediction
                loss_iteration = loss(y_batch_train,
                                      y_pred)

            if cumulPrd is None:
                cumulPrd = np.argmax(y_pred, axis=1)
                cumulY = np.argmax(y_batch_train, axis=1)
            else:
                cumulPrd = np.concatenate((cumulPrd, np.argmax(y_pred, axis=1)), axis=0)
                cumulY = np.concatenate((cumulY, np.argmax(y_batch_train, axis=1)), axis=0)

            grads = tape.gradient(loss_iteration, model.trainable_weights)  # or trainable_variables?
            optimizer.apply_gradients(
                zip(grads, model.trainable_weights))

            tot_loss += loss_iteration
            iterations += 1

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_iteration))
                )
                print("Seen so far: %d samples" % ((step + 1) * self.n_batch))

        train_fm = f1_score(y_true=cumulY, y_pred=cumulPrd, average='weighted')

        cumulY = None
        cumulPrd = None
        tot_loss_val = 0
        iterations_val = 0

        for x_batch_val, y_batch_val in self.val_ds:
            val_pred = model(x_batch_val, training=False)
            if cumulPrd is None:
                cumulPrd = np.argmax(val_pred, axis=1)
                cumulY = np.argmax(y_batch_val, axis=1)
            else:
                cumulPrd = np.concatenate((cumulPrd, np.argmax(val_pred, axis=1)), axis=0)
                cumulY = np.concatenate((cumulY, np.argmax(y_batch_val, axis=1)), axis=0)
                # loss between original variable and prediction
                tot_loss_val += loss(y_batch_val,
                                     val_pred)
                iterations_val += 1

        val_fm = f1_score(y_true=cumulY, y_pred=cumulPrd, average='weighted')
        print("Time taken: %.2fs" % (time.time() - start_time))
        assert isinstance(val_fm, float)
        return model, tot_loss / iterations, tot_loss_val / iterations_val, train_fm, val_fm

    def fit_model(self,
                  model,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=10e-1),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  nb_epoch=10,
                  filepath = '',
                  filename = ''):

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        train_loss_list = val_loss_list = train_metric_list = val_metric_list = [0]
        count = 0

        history = {}
        for e in range(nb_epoch):
            count += 1
            model, train_loss, val_loss, train_metric, val_metric = self.train_classifier(model, optimizer, loss)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_metric_list.append(train_metric)
            val_metric_list.append(val_metric)

            print(
                "Epoch {0}: Training loss {1}& Validation loss {3}& Accuracy training {3}& Accuracy validation {3}".format(
                    str(e),
                    str(round(train_loss.numpy(),3)),
                    str(round(val_loss.numpy(),3)),
                    str(round(train_metric,3)),
                    str(round(val_metric,3))))

            if val_metric >= np.max(val_metric_list):
               model.save(filepath + filename,overwrite = True)



class evaluate_model:
    def __init__(self, x_test, y_test, n_batch=64):
        self.n_batch = n_batch
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        self.test_ds = self.test_ds.batch(self.n_batch)

    def apply_predictions(self, model):

        cumulY = None
        cumulPrd = None
        batch_test_fm = []

        for x_batch_test, y_batch_test in self.test_ds:
            test_pred = model(x_batch_test, training=False)
            if cumulPrd is None:
                cumulPrd = np.argmax(test_pred, axis=1)
                cumulY = np.argmax(y_batch_test, axis=1)

            else:
                cumulPrd = np.concatenate((cumulPrd, np.argmax(test_pred, axis=1)), axis=0)
                cumulY = np.concatenate((cumulY, np.argmax(y_batch_test, axis=1)), axis=0)

            batch_test_fm.append(f1_score(y_true=cumulY, y_pred=cumulPrd, average='weighted'))

        test_fm = f1_score(y_true=cumulY, y_pred=cumulPrd, average='weighted')

        return cumulPrd, test_fm, np.nanmean(batch_test_fm), np.nanstd(batch_test_fm)

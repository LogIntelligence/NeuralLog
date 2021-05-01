import os
import sys
sys.path.append("../")

import pickle
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.nlp import optimization
from sklearn.utils import shuffle

from neurallog.models import NeuralLog
from neurallog import data_loader

log_file = "../data/raw/BGL.log"
emb_dir = "../data/embeddings/BGL"
embed_dim = 768  # Embedding size for each token
max_len = 75

class BatchGenerator(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(self.batch_size)
        dummy = np.zeros(shape=(embed_dim,))
        x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        X = np.zeros((len(x), max_len, embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            if len(x) > max_len:
                x = x[-max_len:]
            x = np.pad(np.array(x), pad_width=((max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            X[item_count] = np.reshape(x, [max_len, embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
        return X[:], Y[:, 0]


def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                    epoch_num, model_name=None):

    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss_object = SparseCategoricalCrossentropy()

    model = NeuralLog(768, loss_object, optimizer)

    # model.load_weights("hdfs_transformer.hdf5")

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        shuffle=True
                        )
    return model


def train(X, Y, epoch_num, batch_size, tx, ty, model_file=None):
    X, Y = shuffle(X, Y)
    n_samples = len(X)
    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    training_generator, num_train_samples = BatchGenerator(train_x, train_y, batch_size), len(train_x)
    validate_generator, num_val_samples = BatchGenerator(val_x, val_y, batch_size), len(val_x)

    print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples,
                                                                                       num_val_samples))

    model = train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                            epoch_num, model_name=model_file)

    return model

if __name__ == '__main__':
    # (x_tr, y_tr), (x_te, y_te) = data_loader.load_Supercomputers(
    #     log_file, train_ratio=0.8, windows_size=20,
    #     step_size=5, e_type='bert', e_name=None, mode='balance')

    with open(os.path.join(emb_dir, "bert-train.pkl"), mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)

    with open(os.path.join(emb_dir, "bert-test.pkl"), mode="rb") as f:
        (x_te, y_te) = pickle.load(f)

    model = train(x_tr, y_tr, 20, 64, x_te, y_te, "bgl_transformer.hdf5")

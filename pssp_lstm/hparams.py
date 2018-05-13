"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf

HPARAMS = tf.contrib.training.HParams(
    num_features=43,
    num_labels=9,
    num_units=300,
    num_layers=3,
    num_dense_units=200,
    dropout=0.5,
    batch_size=50,
    num_epochs=250,
    max_gradient_norm=0.5,
    num_keep_ckpts=9,
)

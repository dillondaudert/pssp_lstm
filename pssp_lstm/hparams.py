"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf

HPARAMS = tf.contrib.training.HParams(
    num_features=43,
    num_labels=9,
    num_units=500,
    num_layers=3,
    num_dense_units=400,
    dropout=0.5,
    batch_size=64,
    num_epochs=100,
    max_gradient_norm=0.5,
    num_keep_ckpts=5,
)

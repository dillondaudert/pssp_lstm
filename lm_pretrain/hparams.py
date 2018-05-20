"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf

hparams = {
   "bdrnn": tf.contrib.training.HParams(
            num_features=43,
            num_inp_labels=22,
            num_tgt_labels=9,
            num_units=500,
            num_layers=2,
            input_proj_size=50,
            num_dense_units=200,
            dropout=0.5,
            batch_size=50,
            num_epochs=10,
            max_gradient_norm=0.5,
            num_keep_ckpts=6,
            model="bdrnn"),
   "lm": tf.contrib.training.HParams(
            num_features=43,
            num_inp_labels=22,
            num_tgt_labels=9,
            num_units=500,
            num_layers=0,
            input_proj_size=50,
            num_dense_units=200,
            dropout=0.5,
            batch_size=50,
            num_epochs=10,
            max_gradient_norm=0.5,
            num_keep_ckpts=6,
            model="lm")
   }

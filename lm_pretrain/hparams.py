"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf
from .bdlm_model import BDLMModel
from .bdrnn_model import BDRNNModel

hparams = {
   "bdlm": tf.contrib.training.HParams(
            num_phyche_features=7,
            num_labels=23,
            num_lm_units=300,
            num_lm_layers=2,
            num_lm_dense_units=200,
            dropout=0.0,
            in_embed_units=10,
            out_embed_units=30,
            batch_size=50,
            num_epochs=200,
            max_gradient_norm=0.5,
            num_keep_ckpts=9,
            model="bdlm",
            Model=BDLMModel),
   }

"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf
from .bdlm_model import BDLMModel
from .bdrnn_model import BDRNNModel

hparams = {
   "bdlm": tf.contrib.training.HParams(
            num_phyche_features=7,
            num_labels=23,
            num_lm_units=1024,
            num_lm_layers=1,
            lm_residual=False,
            cell_type="lstm",
            dropout=0.2,
            recurrent_dropout=0.3,
            in_embed_units=256,
            out_embed_units=256,
            batch_size=50,
            num_epochs=200,
            max_gradient_norm=0.5,
            num_keep_ckpts=11,
            model="bdlm",
            Model=BDLMModel,
            lm_ckpt="",
            train_file="cUR50_train.tfrecords",
            valid_file="cUR50_valid.tfrecords",
            test_file=""),
   }

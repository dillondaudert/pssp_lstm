"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf
from .bdlm_model import BDLMModel
from .bdrnn_model import BDRNNModel

hparams = {
   "bdlm": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_labels=23,
        num_lm_units=256,
        num_lm_layers=1,
        lm_residual=False,
        cell_type="lstm",
        dropout=0.35,
        recurrent_dropout=0.0,
        in_embed_units=256,
        out_embed_units=256,
        batch_size=50,
        num_epochs=200,
        max_gradient_norm=0.5,
        num_keep_ckpts=8,
        eval_step=200,
        model="bdlm",
        Model=BDLMModel,
        train_file="ur50_0*00.tfrecords",
        valid_file="ur50_0*555.tfrecords",
        test_file="",
        bdlm_ckpt="",
        ),
   "bdrnn": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_pssm_features=21,
        num_labels=8,
        num_units=256,
        num_layers=1,
        residual=False,
        cell_type="lstm",
        out_units=256,
        batch_size=50,
        dropout=0.35,
        recurrent_dropout=0.35,
        num_epochs=200,
        max_gradient_norm=1.,
        num_keep_ckpts=8,
        eval_step=60,
        model="bdrnn",
        Model=BDRNNModel,
        train_file="cpdb_train.tfrecords",
        valid_file="cpdb_valid.tfrecords",
        test_file="cpdb513_test.tfrecords",
        ),
   }

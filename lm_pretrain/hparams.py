"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf
from .bdlm_model import BDLMModel
from .bdrnn_model import BDRNNModel

hparams = {
   "bdlm": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_labels=23,
        #num_lm_units=1024,
        num_lm_units=256,
        num_lm_layers=1,
        lm_residual=False,
        cell_type="lstm",
        dropout=0.2,
        recurrent_dropout=0.3,
        in_embed_units=256,
        out_embed_units=256,
        batch_size=50,
        num_epochs=1000,
        max_gradient_norm=0.5,
        num_keep_ckpts=11,
#        eval_step=400, # train steps per eval step
        eval_step=60,
        model="bdlm",
        Model=BDLMModel,
        #train_file="cUR50_train.tfrecords",
        train_file="ur50_0000*.tfrecords",
#        valid_file="cUR50_valid.tfrecords",
        valid_file="ur50_0000*.tfrecords",
        test_file="",
        bdlm_ckpt="",
        ),
   "bdrnn": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_pssm_features=21,
        num_labels=8,
        num_units=300,
        num_layers=1,
        residual=False,
        cell_type="lstm",
        out_units=200,
        batch_size=50,
        dropout=0.3,
        recurrent_dropout=0.3,
        num_epochs=200,
        max_gradient_norm=0.5,
        num_keep_ckpts=7,
        eval_step=60,
        model="bdrnn",
        Model=BDRNNModel,
        train_file="cpdb_train.tfrecords",
        valid_file="cpdb_valid.tfrecords",
        test_file="cpdb513_test.tfrecords",
        ),
   }

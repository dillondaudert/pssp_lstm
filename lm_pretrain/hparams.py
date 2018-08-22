"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf
from .bdlm_model import BDLMModel
from .bdrnn_model import BDRNNModel
from .van_bdrnn_model import VanillaBDRNNModel

alphas = [512, 373, 267, 190, 135, 96]
layers = [1, 2, 4, 8, 16, 32]
i = 0

hparams = {
   "bdlm": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_labels=23,
        num_lm_units=alphas[i],
        num_lm_layers=layers[i],
        lm_residual=True,
        cell_type="lstm",
        dropout=0.,
        recurrent_dropout=0.0,
        in_embed_units=alphas[i] - 7,
        out_embed_units=alphas[i],
        batch_size=50,
        num_epochs=200,
        max_gradient_norm=1.0,
        learning_rate=0.001,
        num_keep_ckpts=12,
        eval_step=200,
        model="bdlm",
        Model=BDLMModel,
        bdlm_ckpt="",
        freeze_bdlm=False,
        file_pattern="ur50_*.tfrecords", # NEW
        file_shuffle_seed=12345, # NEW
        num_train_files=1000,    # NEW
        num_valid_files=10,      # NEW
        ),
   "bdrnn": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_pssm_features=21,
        num_labels=8,
        num_units=256,
        num_layers=4,
        residual=False,
        cell_type="lstm",
        out_units=256,
        batch_size=50,
        learning_rate=0.001,
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
   "van_bdrnn": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_pssm_features=21,
        num_labels=8,
        embed_units=256,
        num_units=256,
        num_layers=2,
        residual=False,
        cell_type="lstm",
        out_units=256,
        batch_size=50,
        learning_rate=0.001,
        dropout=0.35,
        recurrent_dropout=0.35,
        num_epochs=200,
        max_gradient_norm=1.,
        num_keep_ckpts=8,
        eval_step=60,
        model="bdrnn",
        Model=VanillaBDRNNModel,
        train_file="cpdb_train.tfrecords",
        valid_file="cpdb_valid.tfrecords",
        test_file="cpdb513_test.tfrecords",
        ),
   }

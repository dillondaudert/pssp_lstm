"""Hyperparameters for PSSP LSTM"""

import tensorflow as tf
from .bdlm_model import BDLMModel
from .bdrnn_model import BDRNNModel
from .van_bdrnn_model import VanillaBDRNNModel
from .cnn_bdlm_model import CBDLMModel

layers = [0]
alphas = [0]
i = 0

hparams = {
   "cnn_bdlm": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_labels=23,
        num_filters=1024,
        filter_size=7,
        num_units=256,
        num_lm_units=1024,
        num_lm_layers=2,
        dropout=0.0,
        recurrent_state_dropout=0.0,
        recurrent_input_dropout=0.0,
        l2_lambda=0.0,
        l2_alpha=0.0,
        l2_beta=0.,
        batch_size=50,
        num_epochs=16,
        max_gradient_norm=1.0,
        learning_rate=0.5,
        max_patience=5,
        eval_step=1000,
        model="cnn_bdlm",
        Model=CBDLMModel,
        file_pattern="cur50_*.tfrecords",
        file_shuffle_seed=12345,
        num_train_files=1000,
        num_valid_files=10,
        num_gpus=1,
        ),
   "bdrnn": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_pssm_features=21,
        num_labels=8,
        num_units=256,
        num_layers=1,
        filter_size=7,
        residual=False,
        cell_type="lstm",
        out_units=256,
        batch_size=50,
        learning_rate=0.2,
        dropout=0.3,
        recurrent_state_dropout=0.3,
        recurrent_input_dropout=0.,
        num_epochs=300,
        max_gradient_norm=2.,
        max_patience=8,
        eval_step=50,
        model="bdrnn",
        Model=BDRNNModel,
        train_file="cpdb_train.tfrecords",
        valid_file="cpdb_valid.tfrecords",
        test_file="cpdb513_test.tfrecords",
        input_style="out_SEQ", # (1) ELMO_seq, (2) elmo_SEQ, (3) out_SEQ
        ),
   "van_bdrnn": tf.contrib.training.HParams(
        num_phyche_features=7,
        num_pssm_features=21,
        num_labels=8,
        embed_units=256,
        num_units=256,
        num_layers=1,
        residual=False,
        cell_type="lstm",
        out_units=256,
        batch_size=50,
        learning_rate=0.2,
        dropout=0.3,
        recurrent_state_dropout=0.3,
        recurrent_input_dropout=0.0,
        num_epochs=220,
        max_gradient_norm=2.,
        max_patience=8,
        eval_step=50,
        model="van_bdrnn",
        Model=VanillaBDRNNModel,
        train_file="cpdb_train.tfrecords",
        valid_file="cpdb_valid.tfrecords",
        test_file="cpdb513_test.tfrecords",
        ),
   }

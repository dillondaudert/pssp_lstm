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
        #dropout=0.4,
        #recurrent_state_dropout=0.4,
        #recurrent_input_dropout=0.3,
        #l2_lambda=0.0001,
        #l2_alpha=0.001,
        dropout=0.0,
        recurrent_state_dropout=0.0,
        recurrent_input_dropout=0.0,
        l2_lambda=0.0,
        l2_alpha=0.0,
        l2_beta=0.,
        batch_size=50,
        num_epochs=8,
        max_gradient_norm=1.0,
        learning_rate=0.2,
        max_patience=8,
        eval_step=500,
        model="cnn_bdlm",
        Model=CBDLMModel,
        freeze_bdlm=False,
        file_pattern="cur50_*.tfrecords",
        file_shuffle_seed=12345,
        num_train_files=1000,
        num_valid_files=10,
        num_gpus=2,
        ),
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
        max_patience=12,
        eval_step=200,
        model="bdlm",
        Model=BDLMModel,
        freeze_bdlm=False,
        file_pattern="ur50_*.tfrecords",
        file_shuffle_seed=12345,
        num_train_files=1000,
        num_valid_files=10,
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
        max_patience=8,
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
        num_layers=1,
        residual=False,
        cell_type="lstm",
        out_units=256,
        batch_size=50,
        learning_rate=0.1,
        dropout=0.0,
        recurrent_dropout=0.0,
        num_epochs=60,
        max_gradient_norm=1.,
        max_patience=8,
        eval_step=50,
        model="bdrnn",
        Model=VanillaBDRNNModel,
        train_file="cpdb_train.tfrecords",
        valid_file="cpdb_valid.tfrecords",
        test_file="cpdb513_test.tfrecords",
        ),
   }

"""
Model hyperparameter descriptions and helpers.
"""

import argparse as ap

HPARAM_DESCS = {
    "model": (str, "the kind of model to train"),
    "Model": (object, "TODO: consolidate model/Model"),
    "datadir": (str, "the directory where the data files are located"),
    "logdir": (str, "the directory where the logs and model checkpoints should be saved"),
    "logging": (bool, "whether or not performance will be logged during training (disabled by default)"),
    "bdlm_ckpt": (str, "the checkpoint file from which to load a saved BDLM model"),
    "bdrnn_ckpt": (str, "the checkpoint file from which to load a saved BDRNN model"),
    "num_phyche_features": (int, "the number of physicochemical features in the dataset"),
    "num_pssm_features": (int, "the number of features in the position-specific similarity matrices"),
    "input_style": (str, "bdrnn model only: which features to use for input"),
    "num_labels": (int, "the number of target labels in this dataset"),
    "num_units": (int, "the number of units in the recurrent layers of the bdrnn"),
    "num_layers": (int, "the number of layers in the rnn of the bdrnn"),
    "num_lm_units": (int, "the number of units in the recurrent layers of the language model"),
    "num_lm_layers": (int, "the number of layers in the rnn of the language model"),
    "num_filters": (int, "the number of convolution filters to use in the cnn bdlm"),
    "filter_size": (int, "the size of the convolution filters"),
    "l2_lambda": (float, "the strength of the l2 weight regularization to use"),
    "l2_alpha": (float, "the strength of the l2 activity regularization to use"),
    "l2_beta": (float, "the strength of the l2 activity decay regularization to use"),
    "lm_residual": (bool, "whether or not there should be residual connections between layers of the language model"),
    "cell_type": (str, "the type of rnn cell use"),
    "dropout": (float, "the amount of dropout to use for the dense layers of the model"),
    "recurrent_state_dropout": (float, "the amount of dropout to use for the hidden activations of the recurrent layers"),
    "recurrent_input_dropout": (float, "the dropout for the inputs to the recurrent layers"),
    "embed_units": (int, "(BDRNN ONLY) TODO: consolidate with in_embed_units)"),
    "out_units": (int, "(BDRNN ONLY) TODO: consolidate with out_embed_units"),
    "in_embed_units": (int, "the dimensionality of the input embedding"),
    "out_embed_units": (int, "the dimensionality of the output embedding"),
    "batch_size": (int, "the size of the minibatch for training"),
    "num_epochs": (int, "the number of epochs for training"),
    "max_gradient_norm": (float, "the maximum value for the gradient update norm, above which it is truncated"),
    "learning_rate": (float, "the learning rate for training"),
    "max_patience": (int, "the number of non-decreasing eval steps to wait before learning rate is reduced"),
    "eval_step": (int, "the number of training steps to do between each validation pass"),
    "file_pattern": (str, "the pattern of filenames to match for creating a dataset from multiple files.\
                           this is mutually exclusive with [train|valid]_file."),
    "train_file": (str, "the name of the file to use for training. This can also be a pattern to match files."),
    "valid_file": (str, "like train_file, but for validation"),
    "test_file": (str, "like valid_file, but for testing"),
    "freeze_bdlm": (bool, "whether or not the bdlm weights of a model should be frozen during training"),
    "file_shuffle_seed": (int, "a random seed for shuffling the list of files in a dataset"),
    "num_train_files": (int, "the number of files to take from a dataset of files for training"),
    "num_valid_files": (int, "the number of files to take from a dataset of files for validation"),
}

HPARAM_CHOICES = {
    "model": ["bdlm", "bdrnn", "van_bdrnn", "cnn_bdlm"],
    "cell_type": ["lstm", "gru"],
    "input_style": ["ELMO_seq", "elmo_SEQ", "out_SEQ", "base"],
}

def hparams_to_str(hparams):
    outstr = ""
    for hp in HPARAM_DESCS.keys():
        if hp in vars(hparams):
            line = "\t%-20s: %s\n" % (hp, vars(hparams)[hp])
            outstr += line

    return outstr

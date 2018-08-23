"""
Model hyperparameter descriptions and helpers.
"""

import argparse as ap

HPARAM_DESCS = {
    "datadir": (str, "the directory where the data files are located"),
    "logdir": (str, "the directory where the logs and model checkpoints should be saved"),
    "logging": (bool, "whether or not performance will be logged during training (disabled by default)"),
    "bdlm_ckpt": (str, "the checkpoint file from which to load a saved BDLM model"),
    "bdrnn_ckpt": (str, "the checkpoint file from which to load a saved BDRNN model"),
    "num_phyche_features": (int, "the number of physicochemical features in the dataset"),
    "num_pssm_features": (int, "the number of features in the position-specific similarity matrices"),
    "num_labels": (int, "the number of target labels in this dataset"),
    "num_units": (int, "the number of units in the recurrent layers of the bdrnn"),
    "num_layers": (int, "the number of layers in the rnn of the bdrnn"),
    "num_lm_units": (int, "the number of units in the recurrent layers of the language model"),
    "num_lm_layers": (int, "the number of layers in the rnn of the language model"),
    "lm_residual": (bool, "whether or not there should be residual connections between layers of the language model"),
    "cell_type": (str, "the type of rnn cell use"),
    "dropout": (float, "the amount of dropout to use for the dense layers of the model"),
    "recurrent_dropout": (float, "the amount of dropout to use for the hidden activations of the recurrent layers"),
    "variational_dropout": (bool, "whether or not the dropout should be variational for the recurrent layers"),
    "embed_units": (int, "(BDRNN ONLY) TODO: consolidate with in_embed_units)"),
    "out_units": (int, "(BDRNN ONLY) TODO: consolidate with out_embed_units"),
    "in_embed_units": (int, "the dimensionality of the input embedding"),
    "out_embed_units": (int, "the dimensionality of the output embedding"),
    "batch_size": (int, "the size of the minibatch for training"),
    "num_epochs": (int, "the number of epochs for training"),
    "max_gradient_norm": (float, "the maximum value for the gradient update norm, above which it is truncated"),
    "learning_rate": (float, "the learning rate for training"),
    "num_keep_ckpts": (int, "the number of recent model checkpoints to keep at a time"),
    "eval_step": (int, "the number of training steps to do between each validation pass"),
    "model": (str, "the kind of model to train"),
    "Model": ("TODO: consolidate model/Model"),
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
    "model": ["bdlm", "bdrnn", "van_bdrnn"],
    "cell_type": ["lstm", "gru"],
}

def hparams_to_str(hparams):
    outstr = ""
    for hp in HPARAM_DESCS.keys():
        if hp in vars(hparams):
            line = "\t%-20s: %s\n" % (hp, vars(hparams)[hp])
            outstr += line

    return outstr

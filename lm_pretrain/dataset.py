"""Create a tf.data.Dataset input pipeline."""

from pathlib import Path
import tensorflow as tf, numpy as np
from .parsers import cpdb_parser, cUR50_parser
from .lookup import create_lookup_table


def _lm_map_func(hparams, sos_id, eos_id, prot_size):
    """Return a closure for the BDLM with the SOS/EOS ids"""
    def lm_map_func(id, seq_len, seq, phyche):
        prot_eye = tf.eye(prot_size)
        # split characters
        seq = tf.string_split([seq], delimiter="").values
        # map to integers
        seq = tf.cast(hparams.prot_lookup_table.lookup(seq), tf.int32)
        # prepend/append SOS/EOS tokens
        seq_in = tf.concat(([sos_id], seq, [eos_id]), 0)
        if "filter_size" in vars(hparams):
            k = hparams.filter_size
        else:
            k = 1
        # pad zeros to phyche
        phyche_pad = tf.zeros(shape=(k, hparams.num_phyche_features))
        phyche = tf.concat([phyche_pad, phyche, phyche_pad], 0)
        # map to one-hots
        seq_in = tf.nn.embedding_lookup(prot_eye, seq_in)
        seq_out = tf.nn.embedding_lookup(prot_eye, seq)
        # pad zeros to match filters
        if k-1 > 0:
            pad = tf.zeros(shape=(k-1, prot_size))
            seq_in = tf.concat([pad, seq_in, pad], 0)
        return id, seq_len, seq_in, phyche, seq_out
    return lm_map_func

def _bdrnn_map_func(hparams, sos_id, eos_id, prot_size, struct_size):
    """Return a closure for the BDRNN with the SOS/EOS ids"""
    lm_map_func = _lm_map_func(hparams, sos_id, eos_id, prot_size)
    def bdrnn_map_func(id, seq_len, seq, phyche, pssm, ss):
        id, seq_len, seq_in, phyche, seq_out = lm_map_func(id, seq_len, seq, phyche)

        struct_eye = tf.eye(struct_size)
        ss = tf.string_split([ss], delimiter="").values
        ss = tf.cast(hparams.struct_lookup_table.lookup(ss), tf.int32)
        # map to one-hots
        ss = tf.nn.embedding_lookup(struct_eye, ss)
        return id, seq_len, seq_in, phyche, seq_out, pssm, ss
    return bdrnn_map_func

def _from_files(hparams, mode, parser):
    """
    Create a tf.Dataset from a list of files given by a pattern.
    """

    # NOTE: this dataset contains all the files, so it should always be shuffled
    files = tf.data.Dataset.list_files(hparams.file_pattern, shuffle=True, seed=hparams.file_shuffle_seed)

    # take a specified number of files to create the dataset.
    if mode == tf.contrib.learn.ModeKeys.EVAL:
        # if we're evaluating, skip however many files were taken for training
        files = files.skip(hparams.num_train_files)
        files = files.take(hparams.num_valid_files)
    else:
        files = files.take(hparams.num_train_files)

    # id, len, seq: str, phyche(, pssm, ss: str)
    dataset = files.apply(tf.contrib.data.parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(
                  lambda x: parser(x, hparams), num_parallel_calls=1),
                cycle_length=4,
                block_length=10,
                buffer_output_elements=hparams.batch_size,
                prefetch_input_elements=10))

    return dataset

def create_dataset(hparams, mode):
    """
    Create a tf.Dataset from a file.
    Args:
        hparams - Hyperparameters for the dataset
        mode    - the mode, one of tf.contrib.learn.ModeKeys.{TRAIN, EVAL}
    Returns:
        dataset - A tf.data.Dataset object
    """

    # create lookup tables
    hparams.prot_lookup_table = create_lookup_table("prot")
    hparams.prot_reverse_lookup_table = create_lookup_table("prot", reverse=True)
    hparams.struct_lookup_table = create_lookup_table("struct")
    hparams.struct_reverse_lookup_table = create_lookup_table("struct", reverse=True)

    prot_size = tf.cast(hparams.prot_lookup_table.size(), tf.int32)
    struct_size = tf.cast(hparams.struct_lookup_table.size(), tf.int32)

    sos_id = tf.cast(hparams.prot_lookup_table.lookup(tf.constant("SOS")), tf.int32)
    eos_id = tf.cast(hparams.prot_lookup_table.lookup(tf.constant("EOS")), tf.int32)


    batch_size = hparams.batch_size
    # set shuffle and epochs for train/eval
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        shuffle = True
        num_epochs = hparams.num_epochs
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        shuffle = False
        num_epochs = 1
    else:
        print("INFER mode not supported.")
        quit()

    # get parsers and map functions for each kind of dataset
    if hparams.model == "bdlm" or hparams.model == "cnn_bdlm":
        parser = cUR50_parser
        map_fn = _lm_map_func(hparams, sos_id, eos_id, prot_size)
        padded_shapes=(tf.TensorShape([]), # id
                       tf.TensorShape([]), # len
                       tf.TensorShape([None, 23]), # seq
                       tf.TensorShape([None, hparams.num_phyche_features]), # phyche
                       tf.TensorShape([None, 23]), # seq_out
                       )
    elif hparams.model == "bdrnn" or hparams.model == "van_bdrnn":
        parser = cpdb_parser
        map_fn = _bdrnn_map_func(hparams, sos_id, eos_id, prot_size, struct_size)
        padded_shapes=(tf.TensorShape([]), # id
                       tf.TensorShape([]), # len
                       tf.TensorShape([None, 23]), # seq_in
                       tf.TensorShape([None, hparams.num_phyche_features]), # phyche
                       tf.TensorShape([None, 23]), # seq_out
                       tf.TensorShape([None, hparams.num_pssm_features]), # pssm
                       tf.TensorShape([None, hparams.num_labels]), # ss
                       )

    # load file(s) and parse records
    if "file_pattern" in vars(hparams):
        dataset = _from_files(hparams, mode, parser)
    else:
        input_file = hparams.train_file if mode == tf.contrib.learn.ModeKeys.TRAIN \
                                        else hparams.valid_file
        dataset = tf.data.TFRecordDataset(input_file).\
                      map(lambda x: parser(x, hparams), num_parallel_calls=4)

    # filter sequences by length
    dataset = dataset.filter(lambda id, len, *z: tf.logical_and(tf.greater(len, tf.constant(20, dtype=tf.int32)),
                                                                tf.less(len, tf.constant(1040, dtype=tf.int32))))


    # shuffle and repeat
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=50000, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)


    # record transformations
    dataset = dataset.map(map_fn, num_parallel_calls=4)

    if "filter_size" in vars(hparams):
        k = hparams.filter_size
    else:
        k = 1

    # bucketing
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
        lambda id, seq_len, seq_in, phyche, seq_out, *z: seq_len+tf.constant(2*k, dtype=tf.int32),
        [50, 150, 250, 350, # buckets
         450, 550, 650, 850],
        [batch_size, batch_size, batch_size, # all buckets have the
         batch_size, batch_size, batch_size, # the same batch size
         batch_size, batch_size, batch_size],
        padded_shapes=padded_shapes,
        ))

    # prefetch on CPU
    dataset = dataset.prefetch(2)

    return dataset


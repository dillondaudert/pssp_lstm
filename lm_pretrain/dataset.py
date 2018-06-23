"""Create a tf.data.Dataset input pipeline."""

from pathlib import Path
import tensorflow as tf, numpy as np
from .parsers import cpdb_parser, cUR50_parser

def create_dataset(hparams, mode):
    """
    Create a tf.Dataset from a file.
    Args:
        hparams - Hyperparameters for the dataset
        mode    - the mode, one of tf.contrib.learn.ModeKeys.{TRAIN, EVAL, INFER}
    Returns:
        dataset - A tf.data.Dataset object
    """

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        input_file = hparams.train_file
        shuffle = True
        batch_size = hparams.batch_size
        num_epochs = -1
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        input_file = hparams.valid_file
        shuffle = False
        batch_size = hparams.batch_size
        num_epochs = -1 # indefinitely
    else:
        print("INFER mode not supported.")
        quit()

    if hparams.model == "lm":
        parser = cUR50_parser
    elif hparams.model == "bdrnn":
        parser = cpdb_parser

    dataset = tf.data.TFRecordDataset(input_file)

    # parse the records
    # NOTE: id, len, seq: str, phyche(, pssm, ss: str)
    dataset = dataset.map(lambda x:parser(x, hparams), num_parallel_calls=4)

    # create lookup tables for strings
    prot_size = tf.cast(hparams.prot_lookup_table.size(), tf.int32)
    struct_size = tf.cast(hparams.struct_lookup_table.size(), tf.int32)

    # cache results of the parse function
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*100, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    if hparams.model == "lm":
        def lm_map_func(id, seq_len, seq, phyche):
            prot_eye = tf.eye(prot_size)
            # split characters
            seq = tf.string_split([seq], delimiter="").values
            # map to integers
            seq = tf.cast(hparams.prot_lookup_table.lookup(seq), tf.int32)
            # map to one-hots
            seq = tf.nn.embedding_lookup(prot_eye, seq)
            pssm = tf.constant(-1)

            return id, seq_len, seq, phyche, pssm, seq
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche: lm_map_func(id, seq_len, seq, phyche),
                num_parallel_calls=4)

    else:
        def bdrnn_map_func(id, seq_len, seq, phyche, pssm, ss):
            prot_eye = tf.eye(prot_size)
            struct_eye = tf.eye(struct_size)
            # split characters
            seq = tf.string_split([seq], delimiter="").values
            ss = tf.string_split([ss], delimiter="").values
            # map to integers
            seq = tf.cast(hparams.prot_lookup_table.lookup(seq), tf.int32)
            ss = tf.cast(hparams.struct_lookup_table.lookup(ss), tf.int32)
            # map to one-hots
            seq = tf.nn.embedding_lookup(prot_eye, seq)
            ss = tf.nn.embedding_lookup(struct_eye, ss)

            return id, seq_len, seq, phyche, pssm, ss

        dataset = dataset.map(
                lambda id, seq_len, seq, phyche, pssm, ss: bdrnn_map_func(id, seq_len, seq, phyche, pssm, ss),
                num_parallel_calls=4)


    # determine pssm tensorshape
    if hparams.model == "lm":
        pssm_shape = tf.TensorShape([])
        target_shape = tf.TensorShape([None, 23])
    else: #if hparams.model == "bdrnn":
        pssm_shape = tf.TensorShape([None, 21])
        target_shape = tf.TensorShape([None, 10])

    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
        lambda id, seq_len, seq, phyche, pssm, tar: seq_len,
        [50, 150, 250, 350, # buckets
         450, 550, 650],
        [batch_size, batch_size, batch_size, # all buckets have the
         batch_size, batch_size, batch_size, # the same batch size
         batch_size, batch_size],
        padded_shapes=(tf.TensorShape([]), # id
                       tf.TensorShape([]), # len
                       tf.TensorShape([None, 23]), # seq
                       tf.TensorShape([None, hparams.num_phyche_features]), # phyche
                       pssm_shape, # pssm
                       target_shape, # target (ss or seq)
                       )))

    # map to (x, y) tuple for Keras comformability
    dataset = dataset.map(lambda id, seq_len, seq, phyche, pssm, tar: \
                          ((id, seq_len, seq, phyche, pssm),
                          tar))

    # prefetch on CPU
    dataset = dataset.prefetch(2)

    return dataset


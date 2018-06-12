"""Create a tf.data.Dataset input pipeline."""

from pathlib import Path
import tensorflow as tf, numpy as np
from .parsers import cpdb_parser, cUR50_parser
from .lookup import create_lookup_table

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
        num_epochs = hparams.num_epochs
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        input_file = hparams.valid_file
        shuffle = False
        batch_size = hparams.batch_size
        num_epochs = 1
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
    hparams.prot_lookup_table = create_lookup_table("prot")
    hparams.prot_reverse_lookup_table = create_lookup_table("prot", reverse=True)
    hparams.struct_lookup_table = create_lookup_table("struct")
    hparams.struct_reverse_lookup_table = create_lookup_table("struct", reverse=True)
    prot_size = tf.cast(hparams.prot_lookup_table.size(), tf.int32)
    struct_size = tf.cast(hparams.struct_lookup_table.size(), tf.int32)

    if hparams.model == "lm":
        # split characters
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche: \
                        (id, seq_len, tf.string_split([seq], delimiter="").values, phyche),
                num_parallel_calls=4)
        # convert characters to integers
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche: \
                        (id, seq_len, tf.cast(hparams.prot_lookup_table.lookup(seq), tf.int32), phyche),
                num_parallel_calls=4)
        # convert integers to one-hots
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche: \
                        (id, 
                         seq_len, 
                         tf.nn.embedding_lookup(tf.eye(prot_size), seq),
                         phyche),
                num_parallel_calls=4)
        # target is seq itself
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche: \
                        (id, seq_len, seq, phyche, tf.constant(-1), seq),
                num_parallel_calls=4)

    else:
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche, pssm, ss: \
                        (id,
                         seq_len,
                         tf.string_split([seq], delimiter="").values,
                         phyche,
                         pssm,
                         tf.string_split([ss], delimiter="").values),
                num_parallel_calls=4)
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche, pssm, ss: \
                        (id,
                         seq_len,
                         tf.cast(hparams.prot_lookup_table.lookup(seq), tf.int32),
                         phyche,
                         pssm,
                         tf.cast(hparams.struct_lookup_table.lookup(ss), tf.int32)),
                num_parallel_calls=4)
        # convert integers to one-hots
        dataset = dataset.map(
                lambda id, seq_len, seq, phyche, pssm, ss: \
                        (id, 
                         seq_len, 
                         tf.nn.embedding_lookup(tf.eye(prot_size), seq),
                         phyche,
                         pssm,
                         tf.nn.embedding_lookup(tf.eye(struct_size), ss)),
                num_parallel_calls=4)
        
    
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*100, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

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

    # prefetch on CPU
    dataset = dataset.prefetch(2)

    return dataset


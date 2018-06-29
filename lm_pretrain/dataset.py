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

    if "dataset" in hparams:
        dataset = hparams.dataset
        shuffle = False
        num_epochs = hparams.num_epochs
        batch_size = hparams.batch_size
    else:
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

        if hparams.model == "bdlm":
            parser = cUR50_parser
        elif hparams.model == "bdrnn":
            parser = cpdb_parser

        dataset = tf.data.TFRecordDataset(input_file)

        # parse the records
        # NOTE: id, len, seq: str, phyche(, pssm, ss: str)
        dataset = dataset.map(lambda x:parser(x, hparams), num_parallel_calls=4)

    # create lookup tables
    hparams.prot_lookup_table = create_lookup_table("prot")
    hparams.prot_reverse_lookup_table = create_lookup_table("prot", reverse=True)
    hparams.struct_lookup_table = create_lookup_table("struct")
    hparams.struct_reverse_lookup_table = create_lookup_table("struct", reverse=True)

    prot_size = tf.cast(hparams.prot_lookup_table.size(), tf.int32)
    struct_size = tf.cast(hparams.struct_lookup_table.size(), tf.int32)

    # cache results of the parse function
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*100, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)

    sos_id = tf.cast(hparams.prot_lookup_table.lookup(tf.constant("SOS")), tf.int32)
    eos_id = tf.cast(hparams.prot_lookup_table.lookup(tf.constant("EOS")), tf.int32)


    if hparams.model == "bdlm":
        def lm_map_func(id, seq_len, seq, phyche):
            prot_eye = tf.eye(prot_size)
            # split characters
            seq = tf.string_split([seq], delimiter="").values
            # map to integers
            seq = tf.cast(hparams.prot_lookup_table.lookup(seq), tf.int32)
            # prepend/append SOS/EOS tokens
            seq_in = tf.concat(([sos_id], seq, [eos_id]), 0)
            seq_len = seq_len + tf.constant(2, dtype=tf.int32)
            # prepend zeros to phyche
            phyche_pad = tf.zeros(shape=(1, hparams.num_phyche_features))
            phyche = tf.concat([phyche_pad, phyche, phyche_pad], 0)
            # map to one-hots
            seq_in = tf.nn.embedding_lookup(prot_eye, seq_in)
            seq_out = tf.nn.embedding_lookup(prot_eye, seq)

            return id, seq_len, seq_in, phyche, seq_out

        dataset = dataset.map(
                lambda id, seq_len, seq, phyche: lm_map_func(id, seq_len, seq, phyche),
                num_parallel_calls=4)

        dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
            lambda id, seq_len, seq_in, phyche, seq_out: seq_len,
            [50, 150, 250, 350, # buckets
             450, 550, 650],
            [batch_size, batch_size, batch_size, # all buckets have the
             batch_size, batch_size, batch_size, # the same batch size
             batch_size, batch_size],
            padded_shapes=(tf.TensorShape([]), # id
                           tf.TensorShape([]), # len
                           tf.TensorShape([None, 23]), # seq
                           tf.TensorShape([None, hparams.num_phyche_features]), # phyche
                           tf.TensorShape([None, 23]), # seq_out
                           )))


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
            # prepend/append sos/eos tokens and add 2 to length
            seq = tf.concat(([sos_id], seq, [eos_id]), 0)
            seq_len = seq_len + tf.constant(2, dtype=tf.int32)
            # prepend zeros to phyche
            phyche_pad = tf.zeros(shape=(1, hparams.num_phyche_features))
            phyche = tf.concat([phyche_pad, phyche], 0)
            # prepend zeros to pssm
            pssm_pad = tf.zeros(shape=(1, tf.shape(pssm)[1]))
            pssm = tf.concat([pssm_pad, pssm], 0)
            # map to one-hots
            seq = tf.nn.embedding_lookup(prot_eye, seq)
            ss = tf.nn.embedding_lookup(struct_eye, ss)

            return id, seq_len, seq, phyche, pssm, ss

        dataset = dataset.map(
                lambda id, seq_len, seq, phyche, pssm, ss: bdrnn_map_func(id, seq_len, seq, phyche, pssm, ss),
                num_parallel_calls=4)

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
                           tf.TensorShape([None, hparams.num_pssm_features]), # pssm
                           tf.TensorShape([None, 10]), # ss
                           )))

    # prefetch on CPU
    dataset = dataset.prefetch(2)

    return dataset


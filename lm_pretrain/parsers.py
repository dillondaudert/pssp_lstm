"""Parsers for tf records files."""

from pathlib import Path
import tensorflow as tf, numpy as np

def cpdb_parser(record, hparams):
    """
    Parse a CPDB tfrecord Record into a tuple of tensors.
    """

    keys_to_features = {
        "dssp_id": tf.FixedLenFeature([], tf.string),
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq": tf.FixedLenFeature([], tf.string),
        "seq_phyche": tf.VarLenFeature(tf.float32),
        "seq_pssm": tf.VarLenFeature(tf.float32),
        "ss": tf.FixedLenFeature([], tf.string),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    dssp_id = parsed["dssp_id"]
    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = parsed["seq"]
    seq_phyche = tf.sparse_tensor_to_dense(parsed["seq_phyche"])
    seq_phyche = tf.reshape(seq_phyche, [-1, hparams.num_phyche_features])
    seq_pssm = tf.sparse_tensor_to_dense(parsed["seq_pssm"])
    seq_pssm = tf.reshape(seq_pssm, [-1, 21])
    ss = parsed["ss"]

    return dssp_id, seq_len, seq, seq_phyche, seq_pssm, ss


def cUR50_parser(record, hparams):
    """
    Parse a cUR50 tfrecord Record into a tuple of tensors
    """

    keys_to_features = {
        "id": tf.FixedLenFeature([], tf.string),
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq": tf.FixedLenFeature([], tf.string),
        "seq_phyche": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    uniref_id = parsed["id"]
    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = parsed["seq"]
    seq_phyche = tf.sparse_tensor_to_dense(parsed["seq_phyche"])
    seq_phyche = tf.reshape(seq_phyche, [-1, hparams.num_phyche_features])

    return uniref_id, seq_len, seq, seq_phyche


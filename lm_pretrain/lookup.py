"""Utilities for protein and structure strings."""

import tensorflow as tf
from tensorflow.python.ops import lookup_ops


PROT_ALPHABET = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N",
                 "P", "Q", "R", "S", "T", "V", "W", "Y", "X", "SOS", "EOS"]

STRUCT_ALPHABET = ["H", "E", "L", "T", "S", "G", "B", "I"]

def create_lookup_table(vocab, reverse=False):
    """Create a lookup table that turns amino acid or secondary structure
    characters into an integer id.
    Args:
        vocab: One of "prot" or "struct" for amino acids or secondary structures, respectively.
        reverse: Whether or not this table will convert strings to ids (default) or ids to strings.
    Returns:
        A lookup table that maps strings to ids (or ids to strings if reverse==True)
    """

    if vocab == "prot":
        alphabet = PROT_ALPHABET
    elif vocab == "struct":
        alphabet = STRUCT_ALPHABET
    else:
        raise ValueError("Unrecognized value for vocab: %s" % (vocab))

    if not reverse:
        table = lookup_ops.index_table_from_tensor(tf.constant(alphabet))
    else:
        table = lookup_ops.index_to_string_table_from_tensor(tf.constant(alphabet))

    return table


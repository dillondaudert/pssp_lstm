# helper functions for building the Keras model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda

def cut_layer():
    """
    Create a keras Lambda layer that will remove the
    first 2 time steps of a sequence.
    Returns:
        cut_lambda - a Keras.Layer object
    """

    cut_lambda = Lambda(lambda x: x[:, 2:, :])
    return cut_lambda

def rev_layer(seq_lens):
    """
    Create a Keras Lambda layer that will reverse a
    sequence, ignoring any padding.
    Inputs:
        seq_lens - A tensor of lengths with shape (batch,)
    Returns:
        rev_lambda - A Keras.Layer lambda
    """

    rev_lambda = Lambda(lambda x, lens: tf.reverse_sequence(x, tf.reshape(lens, shape=(-1,)), seq_axis=1))
    rev_lambda.arguments = {"lens": seq_lens}
    return rev_lambda


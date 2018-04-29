"""Create a tf.data.Dataset input pipeline."""

from pathlib import Path
import tensorflow as tf, numpy as np

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

    parser = cpdb_parser

    dataset = tf.data.TFRecordDataset(input_file)

    # parse the records
    dataset = dataset.map(lambda x:parser(x, hparams), num_parallel_calls=4)

    # perform the appropriate transformations and return
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*100, count=num_epochs))
    else:
        dataset = dataset.repeat(num_epochs)




    # padded batch to support sequences of multiple lengths
    #dataset = dataset.padded_batch(
    #        batch_size,
    #        padded_shapes=(tf.TensorShape([None, hparams.num_features]),
    #                       tf.TensorShape([None, hparams.num_labels]),
    #                       tf.TensorShape([])))
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
        lambda a, b, seq_len: seq_len,
        [50, 150, 250, 350, # buckets
         450, 550, 650],
        [batch_size, batch_size, batch_size, # all buckets have the
         batch_size, batch_size, batch_size, # the same batch size
         batch_size, batch_size],
        padded_shapes=(tf.TensorShape([None, hparams.num_features]),
                       tf.TensorShape([None, hparams.num_labels]),
                       tf.TensorShape([]))))




    # prefetch on CPU
    dataset = dataset.prefetch(2)

    return dataset

def cpdb_parser(record, hparams):
    """
    Parse a CPDB tfrecord Record into a tuple of tensors.
    """

    keys_to_features = {
        "seq_len": tf.FixedLenFeature([], tf.int64),
        "seq_data": tf.VarLenFeature(tf.float32),
        "label_data": tf.VarLenFeature(tf.float32),
        }

    parsed = tf.parse_single_example(record, keys_to_features)

    seq_len = parsed["seq_len"]
    seq_len = tf.cast(seq_len, tf.int32)
    seq = tf.sparse_tensor_to_dense(parsed["seq_data"])
    label = tf.sparse_tensor_to_dense(parsed["label_data"])
    seq = tf.reshape(seq, [-1, hparams.num_features])
    tgt_outputs = tf.reshape(label, [-1, hparams.num_labels])

    return seq, tgt_outputs, seq_len

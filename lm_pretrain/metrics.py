"""Metrics for evaluation."""

import tensorflow as tf

def streaming_confusion_matrix(labels, predictions, num_classes, weights=None, prefix=""):
    """Calculates a confusion matrix.

    This creates local variables to track the confusion matrix statistics across
    a stream of data.
    Args:
        labels: the ground truth labels, a Tensor of the same shape as predictions
        predictions: the prediction values, a Tensor of shape (?,)
        num_classes: the number of classes for this confusion matrix
        weights: the weight of each prediction (default None)
    Returns:
        confusion: A k x k Tensor representing the confusion matrix, where
            the columns represent the predicted label and the rows represent the
            true label
        update_op: An operation that updates the values in confusion_matrix
            appropriately.
    """

    _confusion = tf.confusion_matrix(labels=labels,
                                     predictions=predictions,
                                     num_classes=num_classes,
                                     weights=weights,
                                     name=prefix+"cm")

    # accumulator for the confusion matrix
    confusion = tf.get_local_variable(name=prefix+"confusion",
                                      shape=[num_classes, num_classes],
                                      dtype=tf.int32,
                                      initializer=tf.zeros_initializer)

    # update op
    update_op = confusion.assign(confusion + _confusion)

    confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                 [1, num_classes, num_classes, 1])

    summary = tf.summary.image(prefix+"confusion_matrix", confusion_image)

    return confusion, update_op

def cm_summary(confusion, num_classes, prefix=""):
    """Create an image summary op for a confusion matrix.
    Returns:
        confusion_summary: Summary of the confusion matrix as an image
    """
    confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                 [1, num_classes, num_classes, 1])

    confusion_summary = tf.summary.image(prefix+"confusion_matrix", confusion_image)
    return confusion_summary

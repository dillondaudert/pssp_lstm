"""Evaluate the model"""
# basic example of training a network end-to-end
from time import process_time
from pathlib import Path
import tensorflow as tf, numpy as np
from .model import create_model

def evaluate(hparams):
    """Evaluate a trained model"""

    eval_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    with eval_tuple.graph.as_default():
        local_initializer = tf.local_variables_initializer()

    print("Evaluating model on %s" % (hparams.valid_file))

    # do evaluation
    eval_tuple.model.saver.restore(eval_tuple.session, hparams.model_ckpt)
    eval_tuple.session.run([eval_tuple.iterator.initializer, local_initializer])
    while True:
        try:
            eval_loss, eval_acc, eval_cm, eval_summary, _ = eval_tuple.model.eval(eval_tuple.session)
            # summary_writer.add_summary(summary, global_step)
        except tf.errors.OutOfRangeError:
            print("Eval Loss: %f, Eval Accuracy: %f" % (eval_loss,
                                                        eval_acc))
            print("Confusion Matrix (true label, predicted label):" )
            print(eval_cm)
            break

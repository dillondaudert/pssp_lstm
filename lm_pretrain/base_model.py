"""
A Base Model class that other models inherit from.
"""

import abc
import tensorflow as tf
from .metrics import streaming_confusion_matrix, cm_summary


class BaseModel(object):

    def __init__(self, hparams, iterator, mode, scope=None):
        self.hparams = hparams
        self.iterator = iterator
        self.mode = mode
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # set initializer
        initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)

        tf.get_variable_scope().set_initializer(initializer)

        inputs = self.iterator.get_next()
        res = self._build_graph(hparams, inputs, mode, scope=scope)

        # Graph losses
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
            self.accuracy = res[2][0]
            self.confusion = res[2][1]
            self.update_metrics = res[3]

        params = tf.trainable_variables()

        # training update ops
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:

            opt = tf.train.AdadeltaOptimizer(learning_rate=1.0,
                                             rho=0.95,
                                             epsilon=1e-06)

            # gradients
            gradients = tf.gradients(self.train_loss,
                                     params)

            # clip by global norm
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)


            self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                              global_step=self.global_step)

            # Summaries
            tf.summary.scalar("grad_norm", gradient_norm, collections=["train"])
            tf.summary.scalar("train_loss", self.train_loss, collections=["train"])
            self.train_summary = tf.summary.merge_all("train")

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # Evaluation summaries
            tf.summary.scalar("eval_loss", self.eval_loss, collections=["eval"])
            tf.summary.scalar("accuracy", self.accuracy, collections=["eval"])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var, collections=["eval"])
            self.eval_summary = tf.summary.merge_all("eval")


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)
        print("BaseModel is expected hparams.lm_ckpt")
        if hparams.lm_ckpt != "":
            self.lm_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="bdlm"))

    @staticmethod
    @abc.abstractmethod
    def _build_graph(hparams, inputs, mode, scope=None):
        """Subclasses must implement this.
        Args:
            hparams: The configuration hyperparameters
            inputs: The result from calling Iterator.get_next()
            scope: The scope for this subgraph
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """
        pass

    def train(self, sess):
        """Do a single training step."""
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.global_step,
                         self.train_summary])


    def train_with_profile(self, sess, writer):
        """Do a single training step, with profiling"""
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        retvals = sess.run([self.update,
                            self.train_loss,
                            self.global_step,
                            self.train_summary], options=run_options,
                                              run_metadata=run_metadata)

        writer.add_run_metadata(run_metadata, "step "+str(retvals[2]), retvals[2])
        return retvals


    def eval(self, sess):
        """Evaluate the model."""
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.accuracy,
                         self.confusion,
                         self.eval_summary,
                         self.update_metrics])
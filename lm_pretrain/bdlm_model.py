"""
A Bidirectional Language Model class.
"""

import tensorflow as tf
from .base_model import BaseModel
from .model_helper import _create_rnn_cell
from .metrics import streaming_confusion_matrix, cm_summary

class BDLMModel(BaseModel):

    def __init__(self, hparams, iterator, mode, scope=None):
        super(BDLMModel, self).__init__(hparams, iterator, mode, scope=scope)



    @staticmethod
    def _build_graph(hparams, inputs, scope=None):
        """Construct the train, evaluation, and inference graphs.
        Args:
            hparams: The hyperparameters for configuration
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        ids, lens, seq_in, phyche, seq_out = inputs

        with tf.variable_scope(scope or "bdlm", dtype=tf.float32) as bdlm_scope:


            in_embed = tf.layers.Dense(units=hparams.in_embed_units,
                                       kernel_initializer=tf.glorot_uniform_initializer(),
                                       use_bias=False,
                                       name="in_embed")(seq_in)

            x = tf.concat([in_embed, phyche], axis=-1)

            fw_cells = _create_rnn_cell(num_units=hparams.num_lm_units,
                                        num_layers=hparams.num_lm_layers,
                                        mode=self.mode)
            bw_cells = _create_rnn_cell(num_units=hparams.num_lm_units,
                                        num_layers=hparams.num_lm_layers,
                                        mode=self.mode)

            #cells.build([None, hparams.num_features]) #hparams.input_proj_size])

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cells,
                    cell_bw=bw_cells,
                    inputs=x,
                    sequence_length=lens,
                    dtype=tf.float32)

            # TODO: Concatenate nonsense
            rnn_out = tf.concat([output_fw, output_bw], axis=-1)

            dense1 = tf.layers.dense(inputs=rnn_out,
                                     units=hparams.num_lm_dense_units,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1,
                                     units=100,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     activation=tf.nn.relu)
            out_embed = tf.layers.Dense(units=hparams.out_embed_units,
                                        kernel_initializer=tf.glorot_normal_initializer(),
                                        name="out_embed")(dense2)


        with tf.variable_scope("lm_out", dtype=tf.float32):
            logits = tf.layers.dense(inputs=out_embed,
                                     units=hparams.num_labels,
                                     use_bias=False)

        # mask out entries longer than target sequence length
        mask = tf.sequence_mask(seq_len, dtype=tf.float32)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=seq_out,
                                                              name="crossent")

        seq_loss = tf.reduce_sum(crossent*mask, axis=1)/tf.cast(lens, tf.float32)
        loss = tf.reduce_sum(seq_loss)/tf.cast(hparams.batch_size, tf.float32)
        # loss = tf.reduce_sum(crossent*mask)/tf.cast(hparams.batch_size, tf.float32)

        metrics = []
        update_ops = []
        if self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # mean eval loss
            loss, loss_update = tf.metrics.mean(values=loss)

            predictions = tf.argmax(input=logits, axis=-1)
            tgt_labels = tf.argmax(input=seq_out, axis=-1)
            acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                  labels=seq_out,
                                                  weights=mask)
            # confusion matrix
            targets_flat = tf.reshape(seq_out, [-1])
            predictions_flat = tf.reshape(predictions, [-1])
            mask_flat = tf.reshape(mask, [-1])
            cm, cm_update = streaming_confusion_matrix(labels=targets_flat,
                                                       predictions=predictions_flat,
                                                       num_classes=hparams.num_labels,
                                                       weights=mask_flat)
            tf.add_to_collection("eval", cm_summary(cm, hparams.num_labels))
            metrics = [acc, cm]
            update_ops = [loss_update, acc_update, cm_update]

        return logits, loss, metrics, update_ops

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
    def _build_graph(hparams, inputs, mode, scope=None):
        """Construct the train, evaluation graphs
        Args:
            hparams: The hyperparameters for configuration
            inputs: An input tuple
            mode: Training/Eval mode
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """
        (x, out_embed), logits, loss, metrics, update_ops = BDLMModel._build_lm_graph(hparams, inputs, mode, scope)
        return logits, loss, metrics, update_ops


    @staticmethod
    def _build_lm_graph(hparams, inputs, mode, scope=None):

        ids, lens, seq_in, phyche, seq_out = inputs

        with tf.variable_scope(scope or "bdlm", dtype=tf.float32) as bdlm_scope:

            in_embed = tf.layers.Dense(units=hparams.in_embed_units,
                                       kernel_initializer=tf.glorot_uniform_initializer(),
                                       use_bias=False,
                                       name="in_embed",
                                       trainable=not hparams.freeze_bdlm)(seq_in)
            in_embed = tf.layers.dropout(inputs=in_embed,
                                         rate=hparams.dropout,
                                         training=mode == tf.contrib.learn.ModeKeys.TRAIN,
                                         name="in_embed_drop")

            x = tf.concat([in_embed, phyche], axis=-1, name="in_embed_all")

            fw_cells = _create_rnn_cell(cell_type=hparams.cell_type,
                                        num_units=hparams.num_lm_units,
                                        num_layers=hparams.num_lm_layers,
                                        mode=mode,
                                        residual=hparams.lm_residual,
                                        recurrent_dropout=hparams.recurrent_dropout,
                                        trainable=not hparams.freeze_bdlm)
            bw_cells = _create_rnn_cell(cell_type=hparams.cell_type,
                                        num_units=hparams.num_lm_units,
                                        num_layers=hparams.num_lm_layers,
                                        mode=mode,
                                        residual=hparams.lm_residual,
                                        recurrent_dropout=hparams.recurrent_dropout,
                                        trainable=not hparams.freeze_bdlm)

            #cells.build([None, hparams.num_features]) #hparams.input_proj_size])

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cells,
                    cell_bw=bw_cells,
                    inputs=x,
                    sequence_length=lens+tf.constant(2, dtype=tf.int32),
                    dtype=tf.float32,
                    swap_memory=False)

            # output_fw/bw are [batch, time, feats]
            output_fw = output_fw[:, :-2, :]
            output_bw = output_bw[:, 2:, :]
            rnn_out = tf.concat([output_fw, output_bw], axis=-1)

            out_embed = tf.layers.Dense(units=hparams.out_embed_units,
                                        kernel_initializer=tf.glorot_normal_initializer(),
                                        name="out_embed",
                                        activation=tf.nn.relu,
                                        trainable=not hparams.freeze_bdlm)(rnn_out)
            out_embed = tf.layers.dropout(inputs=out_embed,
                                          rate=hparams.dropout,
                                          training=mode == tf.contrib.learn.ModeKeys.TRAIN)


        with tf.variable_scope("lm_out", dtype=tf.float32):
            logits = tf.layers.dense(inputs=out_embed,
                                     units=hparams.num_labels,
                                     trainable=not hparams.freeze_bdlm)

        # mask out entries longer than target sequence length
        mask = tf.sequence_mask(lens, dtype=tf.float32)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=seq_out,
                                                              name="crossent")

        seq_loss = tf.reduce_sum(crossent*mask, axis=1)/tf.cast(lens, tf.float32)
        loss = tf.reduce_sum(seq_loss)/tf.cast(hparams.batch_size, tf.float32)
        # loss = tf.reduce_sum(crossent*mask)/tf.cast(hparams.batch_size, tf.float32)

        metrics = []
        update_ops = []
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            # mean eval loss
            loss, loss_update = tf.metrics.mean(values=loss)

            predictions = tf.argmax(input=logits, axis=-1)
            tgt_labels = tf.argmax(input=seq_out, axis=-1)
            acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                  labels=tgt_labels,
                                                  weights=mask)
            # confusion matrix
            targets_flat = tf.reshape(tgt_labels, [-1])
            predictions_flat = tf.reshape(predictions, [-1])
            mask_flat = tf.reshape(mask, [-1])
            cm, cm_update = streaming_confusion_matrix(labels=targets_flat,
                                                       predictions=predictions_flat,
                                                       num_classes=hparams.num_labels,
                                                       weights=mask_flat)
            tf.add_to_collection("eval", cm_summary(cm, hparams.num_labels))
            metrics = [acc, cm]
            update_ops = [loss_update, acc_update, cm_update]

        return (x, out_embed), logits, loss, metrics, update_ops


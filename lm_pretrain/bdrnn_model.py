"""
A Bidirectional RNN Model class.
"""

import tensorflow as tf
from .base_model import BaseModel
from .bdlm_model import BDLMModel
from .model_helper import _create_rnn_cell
from .metrics import streaming_confusion_matrix, cm_summary

class BDRNNModel(BaseModel):

    def __init__(self, hparams, iterator, mode, scope=None):
        super(BDRNNModel, self).__init__(hparams, iterator, mode, scope=scope)

    @staticmethod
    def _build_graph(hparams, inputs, mode, scope=None):
        """Construct the train, evaluation graphs.
        Args:
            hparams: The hyperparameters for configuration
            inputs: A tuple representing an input sample
            mode: Training/eval mode
            scope: The variable scope name for this subgraph
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        ids, lens, seq_in, phyche, seq_out, pssm, ss = inputs

        if hparams.bdlm_ckpt != "":
            (lm_x, lm_out_embed), lm_logits, lm_loss, lm_metrics, lm_update_ops = \
                    BDLMModel._build_lm_graph(hparams.lm_hparams, (ids, lens, seq_in, phyche, seq_out), mode)

            x = tf.concat([lm_x[:, 1:-1, :], lm_out_embed, pssm], axis=-1, name="bdrnn_input")

            if not hparams.train_bdlm:
                x = tf.stop_gradient(x)

        else:
            # strip off extra steps
            seq_in = seq_in[:, 1:-1, :]
            phyche = phyche[:, 1:-1, :]

            in_embed = tf.layers.Dense(units=hparams.in_embed_units,
                                       kernel_initializer=tf.glorot_uniform_initializer(),
                                       use_bias=False,
                                       name="in_embed")(seq_in)
            # TODO: Make it so it's possible to equalize # params with bdlm?
            x = tf.concat([in_embed, phyche, pssm], axis=-1)


        with tf.variable_scope(scope or "bdrnn", dtype=tf.float32) as bdrnn_scope:
            # create bdrnn
            fw_cells = _create_rnn_cell(cell_type=hparams.cell_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        mode=mode,
                                        residual=hparams.residual,
                                        recurrent_dropout=hparams.recurrent_dropout,
                                        as_list=True,
                                        )

            bw_cells = _create_rnn_cell(cell_type=hparams.cell_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        mode=mode,
                                        residual=hparams.residual,
                                        recurrent_dropout=hparams.recurrent_dropout,
                                        as_list=True,
                                        )

            # run bdrnn
            combined_outputs, output_state_fw, output_state_bw = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cells,
                                                                   cells_bw=bw_cells,
                                                                   inputs=x,
                                                                   sequence_length=lens,
                                                                   dtype=tf.float32,
                                                                   scope=bdrnn_scope)
            # dense output layers
            dense1 = tf.layers.dense(inputs=combined_outputs,
                                     units=hparams.out_units,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     activation=tf.nn.relu)
            drop1 = tf.layers.dropout(inputs=dense1,
                                      rate=hparams.dropout,
                                      training=mode==tf.contrib.learn.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=drop1,
                                     units=hparams.num_labels,
                                     use_bias=False,
                                     name="bdrnn_logits")

        # mask out entries longer than target sequence length
        mask = tf.sequence_mask(lens, dtype=tf.float32)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=ss,
                                                              name="crossent")

        seq_loss = tf.reduce_sum(crossent*mask, axis=1)/tf.cast(lens, tf.float32)
        loss = tf.reduce_sum(seq_loss)/tf.cast(hparams.batch_size, tf.float32)

        metrics = []
        update_ops = []
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            # mean eval loss
            loss, loss_update = tf.metrics.mean(values=loss,
                                                name="ss_loss")

            predictions = tf.argmax(input=logits, axis=-1)
            tgt_labels = tf.argmax(input=ss, axis=-1)
            acc, acc_update = tf.metrics.accuracy(predictions=predictions,
                                                  labels=tgt_labels,
                                                  weights=mask,
                                                  name="ss_accuracy")
            # confusion matrix
            targets_flat = tf.reshape(tgt_labels, [-1])
            predictions_flat = tf.reshape(predictions, [-1])
            mask_flat = tf.reshape(mask, [-1])
            cm, cm_update = streaming_confusion_matrix(labels=targets_flat,
                                                       predictions=predictions_flat,
                                                       num_classes=hparams.num_labels,
                                                       weights=mask_flat,
                                                       prefix="ss_")
            tf.add_to_collection("eval", cm_summary(cm, hparams.num_labels, prefix="ss_"))
            metrics = [acc, cm]+lm_metrics
            update_ops = [loss_update, acc_update, cm_update]+lm_update_ops

        return logits, loss, metrics, update_ops


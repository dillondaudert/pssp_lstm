"""
A Bidirectional Language Model with CNN embedding layer.
"""

import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
from .base_model import BaseModel
from .metrics import streaming_confusion_matrix, cm_summary

class CBDLMModel(BaseModel):

    def __init__(self, hparams, iterator, mode, scope=None):
        super(CBDLMModel, self).__init__(hparams, iterator, mode, scope=scope)

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
        outputs, logits, loss, metrics, update_ops = CBDLMModel._build_lm_graph(hparams, inputs, mode, scope)
        return logits, loss, metrics, update_ops


    @staticmethod
    def _build_lm_graph(hparams, inputs, mode, scope=None):

        ids, lens, seq_in, phyche, seq_out = inputs

        x = tf.concat([seq_in, phyche], axis=-1)

        outputs = []

        with tf.variable_scope(scope or "cnn_embed", dtype=tf.float32) as cnn_scope:
            cnn_embed = tf.layers.Conv1D(filters=hparams.num_filters,
                                         kernel_size=hparams.filter_size,
                                         activation=tf.nn.relu,
                                         kernel_regularizer=lambda inp: hparams.l2_lambda*tf.nn.l2_loss(inp),
                                         trainable=not hparams.freeze_bdlm)

            embed_proj = tf.layers.Dense(units=hparams.num_units,
                                         kernel_regularizer=lambda inp: hparams.l2_lambda*tf.nn.l2_loss(inp),
                                         trainable=not hparams.freeze_bdlm)

            z_0 = tf.layers.dropout(inputs=cnn_embed(x),
                                    rate=hparams.dropout,
                                    training=mode == tf.contrib.learn.ModeKeys.TRAIN)
            z_0 = embed_proj(z_0)

            outputs.append([z_0, z_0])


        with tf.variable_scope(scope or "bdlm", dtype=tf.float32) as bdlm_scope:

            _get_cell = lambda name: LSTMCell(name=name,
                                              num_units=hparams.num_lm_units,
                                              num_proj=hparams.num_units,
                                              trainable=not hparams.freeze_bdlm)
            _drop_wrap = lambda cell: tf.nn.rnn_cell.DropoutWrapper(
                    cell=cell,
                    state_keep_prob=1.0-hparams.recurrent_state_dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 1.0,
                    input_keep_prob=1.0-hparams.recurrent_input_dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 1.0,
                    variational_recurrent=True,
                    input_size=tf.TensorShape([1]),
                    dtype=tf.float32)
            fw_cells = []
            bw_cells = []
            # keep track of unwrapped cells so we can get their weights later
            unwrapped_fw_cells = []
            unwrapped_bw_cells = []
            for i in range(hparams.num_lm_layers):
                fw_cell = _get_cell("lstm_fw_%d"%(i))
                bw_cell = _get_cell("lstm_bw_%d"%(i))
                unwrapped_fw_cells.append(fw_cell)
                unwrapped_bw_cells.append(bw_cell)

                fw_cell = _drop_wrap(fw_cell)
                bw_cell = _drop_wrap(bw_cell)

                # create a residual connection around 1st layer
                if i == 0:
                    fw_cell = tf.nn.rnn_cell.ResidualWrapper(fw_cell)
                    bw_cell = tf.nn.rnn_cell.ResidualWrapper(bw_cell)
                # split fw and bw between GPUs
                if hparams.num_gpus == 2:
                    fw_cell = tf.nn.rnn_cell.DeviceWrapper(fw_cell, "/device:GPU:0")
                    bw_cell = tf.nn.rnn_cell.DeviceWrapper(bw_cell, "/device:GPU:1")
                fw_cells.append(fw_cell)
                bw_cells.append(bw_cell)

            # reverse the bw inputs, then reverse all outputs after dynamic_rnn
            outputs[0][1] = tf.reverse_sequence(outputs[0][1],
                                                seq_lengths=lens+tf.constant(2*hparams.filter_size+1, dtype=tf.int32),
                                                seq_axis=1)

            for i in range(hparams.num_lm_layers):
                # get fw / bw outputs for each layer
                input_fw = outputs[-1][0]
                input_bw = outputs[-1][1]

                output_fw, _ = tf.nn.dynamic_rnn(
                        cell=fw_cells[i],
                        inputs=input_fw,
                        sequence_length=lens+tf.constant(2*hparams.filter_size + 1, dtype=tf.int32),
                        dtype=tf.float32)
                output_bw, _ = tf.nn.dynamic_rnn(
                        cell=bw_cells[i],
                        inputs=input_bw,
                        sequence_length=lens+tf.constant(2*hparams.filter_size + 1, dtype=tf.int32),
                        dtype=tf.float32)
                # add weight reg
                unwrapped_fw_cells[i].add_loss(
                        tf.multiply(hparams.l2_lambda, tf.nn.l2_loss(unwrapped_fw_cells[i].weights[0]), name="fw_%d_l2w"%(i))
                        )
                unwrapped_bw_cells[i].add_loss(
                        tf.multiply(hparams.l2_lambda, tf.nn.l2_loss(unwrapped_bw_cells[i].weights[0]), name="bw_%d_l2w"%(i))
                        )
                # add activity reg to last layer
                if i == range(hparams.num_lm_layers)[-1]:
                    unwrapped_fw_cells[i].add_loss(
                            tf.multiply(hparams.l2_alpha, tf.nn.l2_loss(output_fw), name="fw_%d_l2ar"%(i)),
                            inputs=input_fw
                            )
                    unwrapped_bw_cells[i].add_loss(
                            tf.multiply(hparams.l2_alpha, tf.nn.l2_loss(output_bw), name="bw_%d_l2ar"%(i)),
                            inputs=input_bw
                            )

                outputs.append([output_fw, output_bw])

            for i in range(len(outputs)):
                outputs[i][1] =tf.reverse_sequence(outputs[i][1],
                                                    seq_lengths=lens+tf.constant(2*hparams.filter_size+1, dtype=tf.int32),
                                                    seq_axis=1)


        with tf.variable_scope("lm_out", dtype=tf.float32):
            # concat last outputs and feed to softmax
            output_fw = outputs[-1][0][:, :-(hparams.filter_size+1), :]
            output_bw = outputs[-1][1][:, (hparams.filter_size+1):, :]
            rnn_out = tf.concat([output_fw, output_bw], axis=-1)


            rnn_out = tf.layers.dropout(inputs=rnn_out,
                                        rate=hparams.dropout,
                                        training=mode == tf.contrib.learn.ModeKeys.TRAIN)
            logits = tf.layers.dense(inputs=rnn_out,
                                     units=hparams.num_labels,
                                     trainable=not hparams.freeze_bdlm)

        # mask out entries longer than target sequence length
        mask = tf.sequence_mask(lens, dtype=tf.float32)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=seq_out,
                                                              name="crossent")


        seq_loss = tf.reduce_sum(
                tf.reduce_sum(crossent*mask, axis=1)/tf.cast(lens, tf.float32)
                )/tf.cast(hparams.batch_size, tf.float32)
        reg_loss = tf.add_n(tf.losses.get_regularization_losses(), name="reg_loss")

        loss = seq_loss + reg_loss

        metrics = []
        update_ops = []
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            # mean eval loss
            loss, loss_update = tf.metrics.mean(values=loss)
            seq_loss, seq_loss_update = tf.metrics.mean(values=seq_loss)
            reg_loss, reg_loss_update = tf.metrics.mean(values=reg_loss)

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
            update_ops = [loss_update, seq_loss_update, reg_loss_update, acc_update, cm_update]

        return outputs, logits, loss, metrics, update_ops


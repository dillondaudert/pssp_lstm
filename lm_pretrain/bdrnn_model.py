"""
A Bidirectional RNN Model class.
"""

import tensorflow as tf
from .base_model import BaseModel
from .model_helper import _create_rnn_cell
from .metrics import streaming_confusion_matrix, cm_summary

class BDRNNModel(object):

    def __init__(self, hparams, iterator, mode, scope=None):
        super(BDRNNModel, self).__init__(hparams, iterator, mode, scope=scope)
        print("BDRNNModel needs updating to account for both LM and BDRNN inputs/outputs/loss")



    def _build_graph(self, hparams, scope=None):
        """Construct the train, evaluation, and inference graphs.
        Args:
            hparams: The hyperparameters for configuration
            scope: The variable scope name for this subgraph
        Returns:
            A tuple with (logits, loss, metrics, update_ops)
        """

        sample = self.iterator.get_next()

        inputs, tgt_outputs, seq_len = sample

        lm_fw_cell = []
        lm_bw_cell = []
        lm_init_state_fw = []
        lm_init_state_bw = []
        if hparams.pretrained:
            with tf.variable_scope("lm_rnn", dtype=tf.float32):
                # create lm
                with tf.variable_scope("fw", dtype=tf.float32):
                    lm_fw_cell = _create_rnn_cell(num_units=hparams.num_units,
                                                  num_layers=1,
                                                  mode=self.mode)
                    # build the cell so it is in the correct scope
                    # NOTE: this is hard coded
                    lm_fw_cell[0].build([None, hparams.num_features])#hparams.input_proj_size])
                    lm_init_state_fw = _get_initial_state([lm_fw_cell[0].state_size], tf.shape(inputs)[0], "lm")
                with tf.variable_scope("bw", dtype=tf.float32):
                    lm_bw_cell = _create_rnn_cell(num_units=hparams.num_units,
                                                  num_layers=1,
                                                  mode=self.mode)
                    # NOTE: this is hard coded
                    lm_bw_cell[0].build([None, hparams.num_features])#hparams.input_proj_size])
                    lm_init_state_bw = _get_initial_state([lm_bw_cell[0].state_size], tf.shape(inputs)[0], "lm")

                lm_outputs, lm_states = tf.nn.bidirectional_dynamic_rnn(lm_fw_cell[0],
                                                                        lm_bw_cell[0],
                                                                        inputs,
                                                                        sequence_length=seq_len,
                                                                        initial_state_fw=lm_init_state_fw[0],
                                                                        initial_state_bw=lm_init_state_bw[0],
                                                                        dtype=tf.float32)
                # optionally fix the LM weights
                if hparams.fixed_lm:
                    print("Fixing pretrained language models.")
                    lm_outputs = tf.stop_gradient(lm_outputs)
                    lm_outputs = tf.concat([lm_outputs[0], lm_outputs[1]], axis=-1)
                    lm_outputs = tf.layers.dense(lm_outputs,
                                                 20,
                                                 kernel_initializer=tf.glorot_uniform_initializer())
                    lm_outputs = tf.concat([lm_outputs, inputs], axis=-1)


                    #lm_outputs = tf.concat([lm_outputs[0], lm_outputs[1], inputs], axis=-1)
                else:
                    lm_outputs = tf.concat(lm_outputs, axis=-1)



        with tf.variable_scope("bdrnn", dtype=tf.float32) as bdrnn_scope:
            # create bdrnn
            with tf.variable_scope("fw", dtype=tf.float32):
                fw_cells = _create_rnn_cell(num_units=hparams.num_units,
                                            num_layers=hparams.num_layers,
                                            mode=self.mode
                                            )
                init_state_fw = _get_initial_state([cell.state_size for cell in fw_cells],
                                                   tf.shape(inputs)[0], "initial_state_fw")

            with tf.variable_scope("bw", dtype=tf.float32):
                bw_cells = _create_rnn_cell(num_units=hparams.num_units,
                                            num_layers=hparams.num_layers,
                                            mode=self.mode,
                                            )

                init_state_bw = _get_initial_state([cell.state_size for cell in bw_cells],
                                                   tf.shape(inputs)[0], "initial_state_bw")

            # run bdrnn
            combined_outputs, output_state_fw, output_state_bw = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cells,
                                                                   cells_bw=bw_cells,
                                                                   inputs=lm_outputs,
                                                                   sequence_length=seq_len,
                                                                   initial_states_fw=init_state_fw,
                                                                   initial_states_bw=init_state_bw,
                                                                   dtype=tf.float32,
                                                                   scope=bdrnn_scope)
        # outputs is a tuple (output_fw, output_bw)
        # output_fw/output_bw are tensors [batch_size, max_time, cell.output_size]
        # outputs_states is a tuple (output_state_fw, output_state_bw) containing final states for
        # forward and backward rnn

        # concatenate the outputs of each direction
        #combined_outputs = tf.concat([outputs[0], outputs[1]], axis=-1)

        with tf.variable_scope("bdrnn_out", dtype=tf.float32):
            # dense output layers
            dense1 = tf.layers.dense(inputs=combined_outputs,
                                     units=hparams.num_dense_units,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     activation=tf.nn.relu,
                                     use_bias=True)
            drop1 = tf.layers.dropout(inputs=dense1,
                                      rate=hparams.dropout,
                                      training=self.mode==tf.contrib.learn.ModeKeys.TRAIN)
            dense2 = tf.layers.dense(inputs=drop1,
                                     units=hparams.num_dense_units,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     activation=tf.nn.relu,
                                     use_bias=True)
            drop2 = tf.layers.dropout(inputs=dense2,
                                      rate=hparams.dropout,
                                      training=self.mode==tf.contrib.learn.ModeKeys.TRAIN)

            logits = tf.layers.dense(inputs=drop2,
                                     units=hparams.num_labels,
                                     use_bias=False)

        # mask out entries longer than target sequence length
        mask = tf.sequence_mask(seq_len, dtype=tf.float32)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=tgt_outputs,
                                                              name="crossent")

        # divide loss by batch_size * mean(seq_len)
        loss = tf.reduce_sum(crossent*mask)/tf.cast(hparams.batch_size, tf.float32)

        metrics = []
        update_ops = []
        if self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # mean eval loss
            loss, loss_update = tf.metrics.mean(values=loss)

            predictions = tf.argmax(input=logits, axis=-1)
            tgt_labels = tf.argmax(input=tgt_outputs, axis=-1)
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

        return logits, loss, metrics, update_ops


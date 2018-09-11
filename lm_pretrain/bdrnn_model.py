"""
A Bidirectional RNN Model class.
"""

import tensorflow as tf
import collections
from .base_model import BaseModel
from .cnn_bdlm_model import CBDLMModel
from .model_helper import _create_rnn_cell, add_seq_activation_histogram
from .metrics import streaming_confusion_matrix, cm_summary

class BDRNNModel(BaseModel):

    def __init__(self, hparams, iterator, mode, scope=None):
        super(BDRNNModel, self).__init__(hparams, iterator, mode, scope=scope)
        self.bdlm_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="bdlm_"))

    def named_eval(self, sess):
        InputTuple = collections.namedtuple("InputTuple", ["id", "len", "seq_in", "phyche", "seq", "pssm", "ss"])
        OutputTuple = collections.namedtuple("OutputTuple", ["h_0", "h_1", "h_2", "lm_logits"])

        fetches = {"inputs": InputTuple(*self.inputs),
                   "logits": self.logits,
                   "outputs": OutputTuple(*self.outputs)}
        return sess.run(fetches)

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

        # if we aren't fine-tuning the bdlm, set lm_mode to eval
        lm_mode = mode if not hparams.freeze_bdlm else tf.contrib.learn.ModeKeys.EVAL

        outputs, lm_logits, lm_loss, lm_metrics, lm_update_ops = \
                CBDLMModel._build_lm_graph(hparams.lm_hparams, (ids, lens, seq_in, phyche, seq_out), lm_mode, freeze_bdlm=hparams.freeze_bdlm)
        # remove padding for bdlm
        phyche = phyche[:, hparams.lm_hparams.filter_size:-hparams.lm_hparams.filter_size, :]

        with tf.variable_scope("elmo", dtype=tf.float32, reuse=tf.AUTO_REUSE) as elmo_scope:
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(1.0))
            s_task = tf.get_variable("s_task", [len(outputs)], initializer=tf.constant_initializer(1.0))
            s_weights = tf.nn.softmax(s_task, name="s_weights")
            for i in range(len(outputs)):
                tf.summary.scalar("s_weight:h_%d"%i, s_weights[i], collections=["eval"])
            weighted_sum = sum(s_weights[i]*outputs[i] for i in range(len(outputs)))
            elmo = weighted_sum
            gamma_loss = tf.constant(.005)*tf.squared_difference(gamma[0], tf.constant(1.0))

        activation_updates = []

        if hparams.input_style == "ELMO_seq":
            print("ELMO_seq")
            elmo_proj = tf.layers.dense(inputs=elmo,
                                        units=hparams.num_units,
                                        kernel_initializer=tf.glorot_uniform_initializer(),
                                        use_bias=False,
                                        name="elmo_proj")
            seq_dense = tf.layers.dense(inputs=seq_out,
                                        units=25,
                                        kernel_initializer=tf.glorot_uniform_initializer(),
                                        use_bias=False,
                                        name="seq_dense")
            mean_seq_act, mean_seq_act_update = add_seq_activation_histogram(seq_dense, lens, "seq_dense")
            activation_updates += [mean_seq_act_update]
            x = tf.concat([elmo_proj, seq_dense, phyche, pssm], axis=-1)

        elif hparams.input_style == "elmo_SEQ":
            print("elmo_SEQ")
            elmo_proj = tf.layers.dense(inputs=elmo,
                                        units=hparams.lm_hparams.num_labels,
                                        kernel_initializer=tf.glorot_uniform_initializer(),
                                        use_bias=False,
                                        name="elmo_proj")
            seq_dense = tf.layers.dense(inputs=seq_out,
                                        units=hparams.num_units-hparams.lm_hparams.num_labels,
                                        kernel_initializer=tf.glorot_uniform_initializer(),
                                        use_bias=False,
                                        name="seq_dense")
            mean_seq_act, mean_seq_act_update = add_seq_activation_histogram(seq_dense, lens, "seq_dense")
            activation_updates += [mean_seq_act_update]
            x = tf.concat([elmo_proj, seq_dense, phyche, pssm], axis=-1)

        elif hparams.input_style == "out_SEQ":
            print("out_SEQ")
            lm_out = tf.nn.softmax(lm_logits, name="lm_out")
            seq_dense = tf.layers.dense(inputs=seq_out,
                                        units=hparams.num_units-hparams.lm_hparams.num_labels,
                                        kernel_initializer=tf.glorot_uniform_initializer(),
                                        use_bias=False,
                                        name="seq_dense")
            mean_seq_act, mean_seq_act_update = add_seq_activation_histogram(seq_dense, lens, "seq_dense")
            activation_updates += [mean_seq_act_update]


            x = tf.concat([lm_out, seq_dense, phyche, pssm], axis=-1)
        else:
            print("base")
            elmo_proj = tf.layers.dense(inputs=elmo,
                                        units=hparams.num_units,
                                        kernel_initializer=tf.glorot_uniform_initializer(),
                                        use_bias=False)
            x = tf.concat([elmo_proj, pssm], axis=-1)

        x = tf.contrib.layers.layer_norm(inputs=x,
                                         begin_norm_axis=-1,
                                         begin_params_axis=-1)
        mean_x_act, mean_x_act_update = add_seq_activation_histogram(x, lens, "x")
        activation_updates += [mean_x_act_update]

        drop_x = tf.layers.dropout(inputs=x,
                                   rate=hparams.dropout,
                                   training=(mode==tf.contrib.learn.ModeKeys.TRAIN) and hparams.freeze_bdlm)

        #if hparams.freeze_bdlm:
        #    print("Stopping gradients to bdlm.")
        #    x = tf.stop_gradient(x)

        with tf.variable_scope(scope or "bdrnn", dtype=tf.float32) as bdrnn_scope:
            # create bdrnn
            fw_cells = _create_rnn_cell(cell_type=hparams.cell_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        mode=mode,
                                        residual=hparams.residual,
                                        recurrent_state_dropout=hparams.recurrent_state_dropout,
                                        recurrent_input_dropout=hparams.recurrent_input_dropout,
                                        as_list=True,
                                        )

            bw_cells = _create_rnn_cell(cell_type=hparams.cell_type,
                                        num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        mode=mode,
                                        residual=hparams.residual,
                                        recurrent_state_dropout=hparams.recurrent_state_dropout,
                                        recurrent_input_dropout=hparams.recurrent_input_dropout,
                                        as_list=True,
                                        )

            # run bdrnn
            combined_outputs, output_state_fw, output_state_bw = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cells,
                                                                   cells_bw=bw_cells,
                                                                   inputs=drop_x,
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
                                     name="bdrnn_logits")

        # mask out entries longer than target sequence length
        mask = tf.sequence_mask(lens, dtype=tf.float32)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                              labels=ss,
                                                              name="crossent")

        seq_loss = tf.reduce_sum(crossent*mask, axis=1)/tf.cast(lens, tf.float32)
        loss = tf.reduce_sum(seq_loss)/tf.cast(hparams.batch_size, tf.float32)# + \
               #gamma_loss

        if "loss_weights" in vars(hparams):
            print("LM loss weight: %f, PSSP loss weight: %f\n" % (hparams.loss_weights[0], hparams.loss_weights[1]))
            loss = hparams.loss_weights[0]*lm_loss + hparams.loss_weights[1]*loss

        metrics = []
        update_ops = []
        if mode == tf.contrib.learn.ModeKeys.EVAL:
            # mean eval loss
            loss, loss_update = tf.metrics.mean(values=loss,
                                                name="ss_loss")
            #tf.summary.scalar("eval_gamma_loss", gamma_loss, collections=["eval"])

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
            metrics = [acc, cm]
            update_ops = [loss_update, acc_update, cm_update] + activation_updates

        outputs.append(lm_logits)
        return logits, loss, metrics, update_ops, outputs


"""Bidirectional LSTM RNN for protein secondary structure prediction"""
import tensorflow as tf
from custom_rnn.stlstm import STLSTMCell
from collections import namedtuple
from .dataset import create_dataset
from .metrics import streaming_confusion_matrix, cm_summary

ModelTuple = namedtuple('ModelTuple', ['graph', 'iterator', 'model', 'session'])

def create_model(hparams, mode):
    """
    Return a tuple of a tf Graph, Iterator, Model, and Session.
    Args:
        hparams - Hyperparameters; named tuple
        mode    - the tf.contrib.learn mode (TRAIN, EVAL, INFER)
    Returns a ModelTuple(graph, iterator, model, session)
    """

    graph = tf.Graph()

    with graph.as_default():
        dataset = create_dataset(hparams, mode)
        iterator = dataset.make_initializable_iterator()
        model = BDRNNModel(hparams=hparams,
                           iterator=iterator,
                           mode=mode)

    sess = tf.Session(graph=graph)

    modeltuple = ModelTuple(graph=graph, iterator=iterator,
                            model=model, session=sess)

    return modeltuple

class BDRNNModel(object):

    def __init__(self, hparams, iterator, mode, scope=None):
        self.hparams = hparams
        self.iterator = iterator
        self.mode = mode
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # set initializer
        initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)

        tf.get_variable_scope().set_initializer(initializer)

        res = self._build_graph(hparams, scope=scope)

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

        with tf.variable_scope(scope or "dynamic_bdrnn", dtype=tf.float32):

            # create bdrnn
            fw_cells = _create_rnn_cell(num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        mode=self.mode
                                        )

            bw_cells = _create_rnn_cell(num_units=hparams.num_units,
                                        num_layers=hparams.num_layers,
                                        mode=self.mode,
                                        )

            init_state_fw = _get_initial_state([cell.state_size for cell in fw_cells],
                                               tf.shape(inputs)[0], "initial_state_fw")
            init_state_bw = _get_initial_state([cell.state_size for cell in bw_cells],
                                               tf.shape(inputs)[0], "initial_state_bw")


            # run bdrnn
            combined_outputs, output_state_fw, output_state_bw = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cells,
                                                                   cells_bw=bw_cells,
                                                                   inputs=inputs,
                                                                   sequence_length=seq_len,
                                                                   initial_states_fw=init_state_fw,
                                                                   initial_states_bw=init_state_bw,
                                                                   dtype=tf.float32)
            # outputs is a tuple (output_fw, output_bw)
            # output_fw/output_bw are tensors [batch_size, max_time, cell.output_size]
            # outputs_states is a tuple (output_state_fw, output_state_bw) containing final states for
            # forward and backward rnn

            # concatenate the outputs of each direction
            #combined_outputs = tf.concat([outputs[0], outputs[1]], axis=-1)

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

            #stop gradient thru labels by crossent op
            tgt_outputs = tf.stop_gradient(tgt_outputs)

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


def _get_initial_state(state_sizes: list, batch_size, name):
    """
    Create a list of LSTMStateTuple(c, h), with one tuple per layer in state_size. Each state
    vector will have shape [batch_size, cell_size].
    `name` is a prefix for the variable name of the initial states.
    Args:
        state_sizes: A list of RNNCell.state_size values (LSTMStateTuples)

    Example:
        [LSTMStateTuple(c=[batch_size, 300], h=[batch_size, 300]), LSTMStateTuple(c=[batch_size, 300], h=[batch_size, 300])]

    """

    init_states = []

    # for each layer, create a tf variable and tile
    for i, tupl in enumerate(state_sizes):
        c = tf.get_variable(name+"_c_%d"%i, shape=[1, tupl[0]])
        h = tf.get_variable(name+"_h_%d"%i, shape=[1, tupl[1]])
        c_tiled = tf.tile(c, [batch_size, 1])
        h_tiled = tf.tile(h, [batch_size, 1])
        init_states.append(tf.nn.rnn_cell.LSTMStateTuple(c_tiled, h_tiled))

    return init_states

def _create_rnn_cell(num_units, num_layers, mode):
    """Create a list of RNN cells.

    Args:
        num_units: the depth of each unit
        num_layers: the number of cells
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER

    Returns:
        A list of 'RNNCell' instances
    """

    cell_list = []
    for i in range(num_layers):
        single_cell = STLSTMCell(name="stlstm",
                                 num_units=num_units,
                                 st_activation=tf.nn.relu,
                                 st_kernel_initializer=tf.glorot_uniform_initializer(),
                                 st_num_layers=2,
                                 st_residual=True)
        cell_list.append(single_cell)

    return cell_list

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell
from custom_rnn.stlstm import STLSTMCell
from collections import namedtuple
from .dataset import create_dataset

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
    sess = tf.Session(graph=graph)

    with graph.as_default():
        with tf.name_scope("input_pipe"):
            dataset = create_dataset(hparams, mode)
            iterator = dataset.make_initializable_iterator()
        model = hparams.Model(hparams=hparams,
                              iterator=iterator,
                              mode=mode)
        sess.run([tf.tables_initializer()])


    modeltuple = ModelTuple(graph=graph, iterator=iterator,
                            model=model, session=sess)

    return modeltuple

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

def _create_rnn_cell(cell_type, num_units, num_layers, mode, as_list=False):
    """Create a list of RNN cells.

    Args:
        cell_type: the type of RNNCell
        num_units: the depth of each unit
        num_layers: the number of cells
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER
        as_list: return as a list of Cells if True, else as a MultiRNNCell
    Returns:
        A list of 'RNNCell' instances
    """

    if cell_type == "gru":
        Cell = GRUCell
    else:
        Cell = LSTMCell

    cell_list = []
    for i in range(num_layers):
        single_cell = Cell(name=cell_type,
                           num_units=num_units)
#                           initializer=tf.glorot_uniform_initializer)
        cell_list.append(single_cell)

    if not as_list:
        return MultiRNNCell(cell_list)

    return cell_list

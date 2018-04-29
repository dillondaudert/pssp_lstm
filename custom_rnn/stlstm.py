
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class STLSTMCell(rnn_cell_impl.LayerRNNCell):
    _st_kernels=None
    _st_biases=None
    _st_activation=None
    """State transition LSTM, built on top of BasicLSTMCell"""
    def __init__(self, num_units, forget_bias=1.0, st_activation=None,
                 use_st_bias=True, st_kernel_initializer=None,
                 st_bias_initializer=init_ops.zeros_initializer(),
                 st_residual=False, st_num_layers=2,
                 state_is_tuple=True, activation=None, reuse=None,
                 dtype=dtypes.float32, name=None):
        """Initialize the state transition LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          st_activation: an activation function for the transition network
            layers. Default: linear
          use_st_bias: use biases for the transition network layers.
            Default: True
          st_kernel_initializer: an initializer for the fully-connected
            transition kernels. If `None` (default), weights are initialized
            using the default initializer used by `tf.get_variable`
          st_bias_initializer: an initializer for the transition biases.
            Default: zeros_initializer
          st_residual: bool, whether to use residual connections in the
            transition network. Default: `False`
          st_num_layers: int, the number of layers for the state
            transition network
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          dtype: the dtype of the variables of this cell. Default: tf.float32
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
        """
        super(STLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
          logging.warn("%s: Using a concatenated state is slower and will soon be "
                       "deprecated.  Use state_is_tuple=True.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._dtype = dtype
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._st_activation = st_activation
        self._use_st_bias = use_st_bias
        self._st_kernel_initializer = st_kernel_initializer
        self._st_bias_initializer = st_bias_initializer
        self._st_residual = st_residual
        self._st_num_layers = st_num_layers

    @property
    def dtype(self):
        return self._dtype

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        inputs_shape = tensor_shape.TensorShape(inputs_shape)
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        if self._st_num_layers > 0:
            self._st_kernels = [self.add_variable("st_kernel_%d" % i,
                                                  shape=[self._num_units, self._num_units],
                                                  initializer=self._st_kernel_initializer,
                                                  trainable=True) for i in range(self._st_num_layers)]
            if self._use_st_bias:
                self._st_biases = [self.add_variable("st_bias_%d" % i,
                                                     shape=[self._num_units,],
                                                     initializer=self._st_bias_initializer,
                                                     trainable=True) for i in range(self._st_num_layers)]
            else:
                self._st_biases = None
        else:
            self._st_kernels = None
            self._st_biases = None

        self.built = True

    def call(self, inputs, state):
        """State transition long short-term memory cell (STLSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size, 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
          c, h = state
        else:
          c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        out_h = multiply(self._activation(new_c), sigmoid(o))

        def transition(inputs, l):
            """Transform the hidden state via a fully-connected transition layer."""
            outputs = math_ops.matmul(inputs, self._st_kernels[l])
            if self._use_st_bias:
                outputs = nn_ops.bias_add(outputs, self._st_biases[l])
            if self._st_activation is not None:
                outputs = self._st_activation(outputs)
            return outputs

        new_h = out_h
        for l in range(self._st_num_layers):
            new_h = transition(new_h, l)

        if self._st_residual:
            new_h = add(out_h, new_h)

        if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
        else:
          new_state = array_ops.concat([new_c, new_h], 1)
        return out_h, new_state

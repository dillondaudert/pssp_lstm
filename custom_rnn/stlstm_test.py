import stlstm
import tensorflow as tf
import numpy as np
from tensorflow import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops

class STLSTMTest(test.TestCase):

    def test_cell_build(self, *args, **kwargs):
        """Check that state transition variables are created properly."""
        num_units=5
        num_layers=2
        inputs_shape=[2,8]
        st_kernel_initializer=init_ops.identity_initializer()

        # NOTE: test building with num_layers=0
        with self.test_session():
            cell = stlstm.STLSTMCell(num_units,
                                     st_num_layers=0)

            self.assertIsNone(cell._st_kernels)
            self.assertIsNone(cell._st_biases)

        with self.test_session():
            cell = stlstm.STLSTMCell(num_units,
                                     st_kernel_initializer=st_kernel_initializer,
                                     st_num_layers=num_layers)
            cell.build(inputs_shape)
            variables.global_variables_initializer().run()
            # check cell._st_kernels/biases
            # NOTE: check length of new variable arrays
            self.assertEqual(num_layers, len(cell._st_kernels))
            self.assertEqual(len(cell._st_kernels), len(cell._st_biases))

            # NOTE: check sizes of new variable arrays
            for layer in range(num_layers):
                self.assertAllEqual(cell._st_kernels[layer].shape, [num_units, num_units])
                self.assertEqual(cell._st_biases[layer].shape[0], num_units)

            # NOTE: check variables are initialied correctly
            for layer in range(num_layers):
                self.assertAllEqual(cell._st_kernels[layer].eval(), np.eye(num_units))
                self.assertAllEqual(cell._st_biases[layer].eval(), np.zeros(num_units))

    def test_cell_call(self, *args, **kwargs):
        """Test that the state transition works properly on hidden-to-hidden states."""
        batch_size = 2
        num_units = 3
        with self.test_session() as sess:
            cell_0layer = stlstm.STLSTMCell(num_units, st_num_layers=0)
            cell_1layer = stlstm.STLSTMCell(num_units, st_num_layers=1,
                                            st_kernel_initializer=init_ops.identity_initializer())

            init_state0 = cell_0layer.zero_state(batch_size, dtype=dtypes.float32)
            init_state1 = cell_1layer.zero_state(batch_size, dtype=dtypes.float32)

            inputs = random_ops.random_uniform([batch_size, 5])
            h0, st0 = cell_0layer(inputs, init_state0)
            h1, st1 = cell_1layer(inputs, init_state1)
            variables.global_variables_initializer().run()

            out0, newst0 = sess.run([h0, st0])
            out1, newst1 = sess.run([h1, st1])

            # NOTE: assert that a stlstm with 0 st layers output is the same as the new hidden state
            self.assertAllEqual(out0, newst0[1])
            # NOTE: assert the same for 1 layer st with an identity kernel and zero biases
            self.assertAllEqual(out1, newst1[1])

    def test_cell_residual(self, *args, **kwargs):
            num_units = 3
            # test residual and bias
            # NOTE: test that 2 layer identity kernel with ones biases, new hidden state = output + output + 2
            with self.test_session() as sess:
                cell = stlstm.STLSTMCell(num_units, st_num_layers=2,
                                         st_kernel_initializer=init_ops.identity_initializer(),
                                         st_bias_initializer=init_ops.constant_initializer(1.),
                                         st_residual=True)
                init_state = cell.zero_state(2, dtype=dtypes.float32)
                inputs = random_ops.random_uniform([2, 5])
                h, st = cell(inputs, init_state)
                variables.global_variables_initializer().run()

                out, newst = sess.run([h, st])

                self.assertAllClose(2.*out+2, newst[1])


if __name__ == '__main__':
    tf.test.main()

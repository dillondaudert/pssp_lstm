# test the model_helper functions
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, concatenate, SimpleRNN, Bidirectional
from .model_helper import cut_layer, rev_layer

class TestModelHelperFunctions(unittest.TestCase):

    data = np.array([[[-1], [1], [2], [3], [4], [-2], [0], [0]],
                     [[-1], [1], [2], [3], [-2], [0], [0], [0]]])

    targets = np.array([[[-1, 2], [1, 3], [2, 4], [3, -2], [0, 0], [0, 0]],
                        [[-1, 2], [1, 3], [2, -2], [0, 0], [0, 0], [0, 0]]])

    lens = np.array([[6], [5]], dtype=np.int32)

    def setUp(self):
        self.x = Input(shape=(None, 1))
        self.x_lens = Input(shape=(1,), dtype="int32")

    def tearDown(self):
        keras.backend.clear_session()

    def test_cut_layer(self):
        # build model with cut layer
        cut_lam = cut_layer()
        y = cut_lam(self.x)
        model = Model(inputs=[self.x], outputs=[y])
        model.compile(loss="mse", optimizer="rmsprop")

        preds = model.predict([self.data])

        self.assertTrue(np.array_equal(self.data[:, 2:, :], preds))

    def test_rev_layer(self):
        rev_lam = rev_layer(self.x_lens)
        y = rev_lam(self.x)
        model = Model(inputs=[self.x, self.x_lens], outputs=[y])
        model.compile(loss="mse", optimizer="rmsprop")

        preds = model.predict([self.data, self.lens])

        rev_data = np.array([[[-2], [4], [3], [2], [1], [-1], [0], [0]],
                             [[-2], [3], [2], [1], [-1], [0], [0], [0]]])

        self.assertTrue(np.array_equal(rev_data, preds))

    def test_rev_twice(self):
        rev_lam = rev_layer(self.x_lens)
        y = rev_lam(rev_lam(self.x))
        model = Model(inputs=[self.x, self.x_lens], outputs=[y])
        model.compile(loss="mse", optimizer="rmsprop")

        preds = model.predict([self.data, self.lens])

        self.assertTrue(np.array_equal(self.data, preds))

    def test_rev_cut(self):
        cut_lam = cut_layer()
        rev_lam = rev_layer(self.x_lens)
        rev_lam_2 = rev_layer(self.x_lens - tf.constant(np.array([2]), dtype=tf.int32))

        #reverse, cut, reverse
        reverse = rev_lam(self.x)
        cut = cut_lam(reverse)
        y = rev_lam_2(cut)
        model = Model(inputs=[self.x, self.x_lens], outputs=[y])
        model.compile(loss="mse", optimizer="rmsprop")

        cut_data = np.array([[[-1], [1], [2], [3], [0], [0]],
                            [[-1], [1], [2], [0], [0], [0]]])
        preds = model.predict([self.data, self.lens])

        self.assertTrue(np.array_equal(cut_data, preds))

    def test_rev_cut_rnn(self):
        cut_lam = cut_layer()
        rev_lam = rev_layer(self.x_lens)
        unrev_lam = rev_layer(self.x_lens - tf.constant(2, dtype=tf.int32))

        rnn = Bidirectional(SimpleRNN(units=1,
                                      activation="linear",
                                      kernel_initializer="ones",
                                      recurrent_initializer="zeros",
                                      return_sequences=True),
                            merge_mode=None)
        fw, bw = rnn(self.x)
        # cut bw
        bw_cut = cut_lam(bw)
        # rev, cut, unrev fw
        fw_rev = rev_lam(fw)
        fw_cut = cut_lam(fw_rev)
        fw_unrev = unrev_lam(fw_cut)

        y = concatenate([fw_unrev, bw_cut], axis=-1)
        model = Model(inputs=[self.x, self.x_lens], outputs=[y])
        model.compile(loss="mse", optimizer="rmsprop")

        preds = model.predict([self.data, self.lens])

        self.assertTrue(np.array_equal(self.targets, preds))

    def test_rev_cut_rnn_dataset(self):
        cut_lam = cut_layer()
        rev_lam = rev_layer(self.x_lens)
        unrev_lam = rev_layer(self.x_lens - tf.constant(2, dtype=tf.int32))

        rnn = Bidirectional(SimpleRNN(units=1,
                                      activation="linear",
                                      kernel_initializer="ones",
                                      recurrent_initializer="zeros",
                                      return_sequences=True),
                            merge_mode=None)
        fw, bw = rnn(self.x)
        # cut bw
        bw_cut = cut_lam(bw)
        # rev, cut, unrev fw
        fw_rev = rev_lam(fw)
        fw_cut = cut_lam(fw_rev)
        fw_unrev = unrev_lam(fw_cut)

        y = concatenate([fw_unrev, bw_cut], axis=-1)
        model = Model(inputs=[self.x, self.x_lens], outputs=[y])
        model.compile(loss="mse", optimizer="rmsprop")

        dataset = tf.data.Dataset.from_tensor_slices((self.data, self.lens, self.targets))
        dataset = dataset.map(lambda x, l, t: ((x, l), t))
        dataset = dataset.repeat(100)
        dataset = dataset.batch(2)

        preds = model.predict(dataset, steps=1)

        self.assertTrue(np.array_equal(self.targets, preds))



if __name__ == "__main__":
    unittest.main()

# test the dataset pipeline
import unittest
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_tensor
import numpy as np
from .dataset import _from_files, _lm_map_func, _bdrnn_map_func
from .parsers import cpdb_parser, cUR50_parser

tf.logging.set_verbosity(0)

class TestMapFuncs(unittest.TestCase):
    """
    Test the dataset map functions.
    """

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()
        cls.sess = tf.Session(graph=cls.graph)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cls.alpha = ["A", "B", "C", "D"]
        with cls.graph.as_default():
            lookup = index_table_from_tensor(tf.constant(cls.alpha))
            cls.sess.run([tf.tables_initializer()])
            cls.bdlm_hparams = tf.contrib.training.HParams(
                model="bdlm",
                prot_lookup_table=lookup,
                struct_lookup_table=lookup,
                num_phyche_features=1,
                )
            cls.cnn_bdlm_hparams = tf.contrib.training.HParams(
                model="cnn_bdlm",
                prot_lookup_table=lookup,
                struct_lookup_table=lookup,
                num_phyche_features=1,
                filter_size=3,
                )
            cls.table_size = tf.cast(lookup.size(), tf.int32)
            cls.sos_id = tf.cast(lookup.lookup(tf.constant("C")), tf.int32)
            cls.eos_id = tf.cast(lookup.lookup(tf.constant("D")), tf.int32)

            x1 = (tf.constant(0),
                  tf.constant(3),
                  tf.constant("AAB"),
                  .25*tf.ones((3, 1), dtype=tf.float32),
                  .5*tf.ones((3, 1), dtype=tf.float32)
                 )
            x2 = (tf.constant(1),
                  tf.constant(6),
                  tf.constant("BBABAB"),
                  .25*tf.ones((6, 1), dtype=tf.float32),
                  .5*tf.ones((6, 1), dtype=tf.float32),
                 )
            cls.xs = [x1, x2]


    def test_bdlm_mapfn(self):
        y1 = (0,
              3,
              np.array([[0., 0., 1., 0.], # sos (C)
                        [1., 0., 0., 0.], # A
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [0., 0., 0., 1.]]),# eos (D)
              np.array([[0.],
                        [.25],
                        [.25],
                        [.25],
                        [0.]]),
              np.array([[1., 0., 0., 0.], # A
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.]]),# B
              )
        y2 = (1,
              6,
              np.array([[0., 0., 1., 0.], # sos (C)
                        [0., 1., 0., 0.], # B
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [0., 0., 0., 1.]]),# eos (D)
              np.array([[0.],
                        [.25],
                        [.25],
                        [.25],
                        [.25],
                        [.25],
                        [.25],
                        [0.]]),
              np.array([[0., 1., 0., 0.], # B
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.]]),# B
              )
        ys = [y1, y2]

        with self.graph.as_default():
            lm_map_fn = _lm_map_func(self.bdlm_hparams, self.sos_id, self.eos_id, self.table_size)

            for i, x in enumerate(self.xs):
                id, len, seq_in, phyche, seq_out = self.sess.run(lm_map_fn(x[0], x[1], x[2], x[3]))
                self.assertEqual(ys[i][0], id)
                self.assertEqual(ys[i][1], len)
                self.assertTrue(np.array_equal(ys[i][2], seq_in))
                self.assertTrue(np.allclose(ys[i][3], phyche))
                self.assertTrue(np.array_equal(ys[i][4], seq_out))

    def test_cnn_bdlm_mapfn(self):
        y1 = (0,
              3,
              np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 1., 0.], # sos (C)
                        [1., 0., 0., 0.], # A
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [0., 0., 0., 1.],# eos (D)
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]]),
              np.array([[0.],
                        [0.],
                        [0.],
                        [.25],
                        [.25],
                        [.25],
                        [0.],
                        [0.],
                        [0.]]),
              np.array([[1., 0., 0., 0.], # A
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.]]),# B
              )
        y2 = (1,
              6,
              np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 1., 0.], # sos (C)
                        [0., 1., 0., 0.], # B
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [0., 0., 0., 1.],# eos (D)
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]]),
              np.array([[0.],
                        [0.],
                        [0.],
                        [.25],
                        [.25],
                        [.25],
                        [.25],
                        [.25],
                        [.25],
                        [0.],
                        [0.],
                        [0.]]),
              np.array([[0., 1., 0., 0.], # B
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.], # B
                        [1., 0., 0., 0.], # A
                        [0., 1., 0., 0.]]),# B
              )
        ys = [y1, y2]

        with self.graph.as_default():
            lm_map_fn = _lm_map_func(self.cnn_bdlm_hparams, self.sos_id, self.eos_id, self.table_size)

            for i, x in enumerate(self.xs):
                id, len, seq_in, phyche, seq_out = self.sess.run(lm_map_fn(x[0], x[1], x[2], x[3]))
                self.assertEqual(ys[i][0], id)
                self.assertEqual(ys[i][1], len)
                self.assertTrue(np.array_equal(ys[i][2], seq_in))
                self.assertTrue(np.allclose(ys[i][3], phyche))
                self.assertTrue(np.array_equal(ys[i][4], seq_out))




if __name__ == "__main__":
    unittest.main()

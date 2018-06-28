# test the dataset pipeline
import unittest
import tensorflow as tf
import numpy as np
from .dataset import create_dataset
from .lookup import create_lookup_table

class TestDatasetPipeline(unittest.TestCase):

    hparams = tf.contrib.training.HParams(
            num_phyche_features=1,
            num_pssm_features=1,
            batch_size=2,
            num_epochs=100,
            prot_lookup_table=None,
            struct_lookup_table=None,
            dataset=None,
            model=None)

    ids = ["0", "1"]
    lens = [6, 4]
    seqs = ["ACDEFG", "ACDE"]
    phyche = [np.ones(shape=(6, 1)), np.ones(shape=(4, 1))]
    pssm = [np.ones(shape=(6, 1))*.5, np.ones(shape=(4, 1))*.5]
    ss = ["HELTSG", "HELT"]
    def data_gen(self):
        for i in range(2):
            yield (self.ids[i],
                   self.lens[i],
                   self.seqs[i],
                   self.phyche[i],
                   self.pssm[i],
                   self.ss[i])

    def setUp(self):
        self.sess = tf.Session()
        self.dataset = tf.data.Dataset.from_generator(
                self.data_gen,
                output_types=(tf.string, tf.int32, tf.string, tf.float32, tf.float32, tf.string),
                output_shapes=(tf.TensorShape([]),
                               tf.TensorShape([]),
                               tf.TensorShape([]),
                               tf.TensorShape([None, 1]),
                               tf.TensorShape([None, 1]),
                               tf.TensorShape([])))

        self.hparams.prot_lookup_table = create_lookup_table("prot")
        self.hparams.struct_lookup_table = create_lookup_table("struct")
        self.sess.run(tf.tables_initializer())


    def tearDown(self):
        self.sess.close()
        tf.reset_default_graph()


    def test_dataset_from_gen(self):
        iterator = self.dataset.make_one_shot_iterator()
        x = iterator.get_next()
        next(self.data_gen())
        self.sess.run(x)

    def test_dataset_lm(self):
        self.dataset = self.dataset.map(lambda ids, lens, seqs, phyche, pssm, ss: (ids, lens, seqs, phyche))
        self.hparams.dataset = self.dataset
        self.hparams.model = "lm"
        dataset = create_dataset(self.hparams, tf.contrib.learn.ModeKeys.EVAL)
        iterator = dataset.make_initializable_iterator()
        x = iterator.get_next()
        self.sess.run(iterator.initializer)
        (_id, _len, _seq, _phyche, __), _ = self.sess.run(x)
        print("SEQ:")
        print(_seq)
        print("PHYCHE:")
        print(_phyche)

#        target_seq = np.array([[[

#        self.assertTrue(np.array_equal( , _seq))




if __name__ == "__main__":
    unittest.main()

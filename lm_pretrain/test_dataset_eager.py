# test the dataset pipeline
import unittest
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_tensor
import numpy as np
from .dataset import _from_files, _lm_map_func, _bdrnn_map_func
from .parsers import cpdb_parser, cUR50_parser

class TestFromFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cls.datadir = Path(Path.home(), "data", "test_data")
        tf.enable_eager_execution()
        cls.hparams = tf.contrib.training.HParams(
            file_pattern=str(Path(cls.datadir, "ur50_*.tfrecords")),
            file_shuffle_seed=12345,
            num_train_files=1,
            num_valid_files=1,
            batch_size=1,
            num_phyche_features=7,
            )
        cls.tr_mode = tf.contrib.learn.ModeKeys.TRAIN
        cls.ev_mode = tf.contrib.learn.ModeKeys.EVAL

    def test_from_files_train(self):
        train_dataset1 = _from_files(self.hparams, self.tr_mode, cUR50_parser)
        train_dataset2 = _from_files(self.hparams, self.tr_mode, cUR50_parser)

        iter1 = train_dataset1.make_one_shot_iterator()
        iter2 = train_dataset2.make_one_shot_iterator()

        for (a, b) in zip(iter1, iter2):
            self.assertEqual(str(a[0].numpy()), str(b[0].numpy()))


    def test_from_files_valid(self):
        train_dataset = _from_files(self.hparams, self.tr_mode, cUR50_parser)
        train_iter = train_dataset.make_one_shot_iterator()
        # valid_dataset skips 1 and reads 1 (from self.hparams)
        valid_dataset = _from_files(self.hparams, self.ev_mode, cUR50_parser)
        valid_iter = valid_dataset.make_one_shot_iterator()
        # train_dataset2 reads 2
        hparams = tf.contrib.training.HParams(
            file_pattern=str(Path(self.datadir, "ur50_*.tfrecords")),
            file_shuffle_seed=12345,
            num_train_files=2,
            num_valid_files=1,
            batch_size=1,
            num_phyche_features=7,
            )
        train_dataset2 = _from_files(hparams, self.tr_mode, cUR50_parser)
        train_iter2 = train_dataset2.make_one_shot_iterator()

        # NOTE: so the IDs in valid_ds should be a subset of the IDs from train_ds
        #       and the intersection should be equal to the set of valid_ds IDs.
        train_ids = set((str(x[0].numpy()) for x in train_iter))
        valid_ids = set((str(x[0].numpy()) for x in valid_iter))
        train2_ids = set((str(x[0].numpy()) for x in train_iter2))

        # this checks that the valid dataset skips num_train_files
        self.assertTrue(valid_ids <= train2_ids)

        # this checks that the train dataset reads the same file for both
        self.assertTrue(train_ids <= train2_ids)
        self.assertTrue((train2_ids - valid_ids) == train_ids)


if __name__ == "__main__":
    unittest.main()

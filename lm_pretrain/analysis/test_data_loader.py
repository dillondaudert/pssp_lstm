
import os
import unittest as ut
import pandas as pd
import numpy as np
from .data_loader import load_data

class TestDataLoader(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Write dummy data to a pickle file to test data_loader
        """
        cls.test_file_1 = "/tmp/test_data_loader_dummy_1.pkl"
        cls.test_file_2 = "/tmp/test_data_loader_dummy_2.pkl"
        cls.in_cols = ["file", "id", "len", "seq", "phyche", "pssm", "logits",
                       "ss", "h_0", "h_1", "h_2", "lm_logits"]
        cls.out_cols = ["dataset", "id", "len", "position", "amino",
                        "phyche", "pssm", "logits", "ss", "h_0", "h_1", "h_2",
                        "lm_logits"]

        seq = np.array([[0., 0., 1.],
                        [1., 0., 0.]])
        phyche = np.array([[0., 0.],  # phyche
                           [1., 0.]])
        pssm = np.array([[0., 0., .8],  # pssm
                         [.8, 0., 0.]])
        logits = np.array([[0.1, 0., 0.9],  # logits
                           [0.9, 0., 0.1]])
        ss = np.array([[0., 0., 1.],  # ss
                       [1., 0., 0.]])
        h_0 = np.array([[0., 0., 1., 0.],
                        [1., 0., 0., 0.]])
        h_1 = np.array([[0., 0., 1., 0.],
                        [1., 0., 0., 0.]])
        h_2 = np.array([[0., 0., 1., 0.],  # h_2
                        [1., 0., 0., 0.]])
        lm_logits = np.array([[0., 0., 1.],  # lm_logits
                             [1., 0., 0.]])

        ex_1_in = ("dummy_train.tfrecords", # file
                   "id1",                   # id
                   2,                       # len
                   seq,
                   phyche,
                   pssm,
                   logits,
                   ss,
                   h_0,
                   h_1,
                   h_2,
                   lm_logits,
                  )
        ex_1_out = [tuple(["train", ex_1_in[1], ex_1_in[2], j] + [ex_1_in[i][j, :] for i in range(3, len(ex_1_in))]) for j in range(2)]

        in_df = pd.DataFrame.from_records(data=[ex_1_in], columns=cls.in_cols)
        # write to file
        in_df.to_pickle(cls.test_file_1)

        cls.out_df = pd.DataFrame.from_records(data=ex_1_out, columns=cls.out_cols)

    def test_load_data(self):
        df = load_data(self.test_file_1)
        self.assertTrue(np.allclose(df["logits"].iloc[0], self.out_df["logits"].iloc[0]))


    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_file_1)



if __name__ == "__main__":
    ut.main()

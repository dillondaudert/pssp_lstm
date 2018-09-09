
import unittest as ut
import pandas as pd
import numpy as np
from .metrics import *

class TestMetrics(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.DataFrame(
                data={"y_true": [np.array([1., 0.]),
                                 np.array([0., 1.]),
                                 np.array([1., 0.]),
                                 np.array([1., 0.]),
                                 np.array([0., 1.]),
                                 np.array([1., 0.])],
                      "logits": [np.array([.49, .51]),
                                 np.array([.1, .9]),
                                 np.array([.9, .1]),
                                 np.array([.9, .1]),
                                 np.array([.51, .49]),
                                 np.array([.9, .1])],
                      "lens": [4, 4, 4, 4, 2, 2],
                      "res_pos": [0, 1, 2, 3, 0, 1],
                      })

        cls.classes = ["Class A", "Class B"]

        cls.accuracy = pd.DataFrame(
                data={"accuracy": [.75, .5, 4./6.]},
                index=["Class A", "Class B", "Total"]
                )



    def test_cross_entropy_loss(self):
        pass

    def test_cross_entropy_loss_vs_len(self):
        pass

    def test_cross_entropy_loss_vs_pos(self):
        pass

    def test_class_accuracy(self):
        y_true = self.test_data.y_true.apply(np.argmax)
        y_pred = self.test_data.logits.apply(np.argmax)
        self.assertEqual(self.accuracy.loc["Class A"][0],
                         _class_accuracy(y_true, y_pred, 0))
        self.assertEqual(self.accuracy.loc["Class B"][0],
                         _class_accuracy(y_true, y_pred, 1))

    def test_accuracy(self):
        y_true = self.test_data.y_true.apply(np.argmax)
        y_pred = self.test_data.logits.apply(np.argmax)
        acc = accuracy(y_true, y_pred, classes=self.classes)
        res = all(np.isclose(acc.values, self.accuracy.values)) \
                and list(acc.index) == self.classes+["Total"]
        self.assertTrue(res)

    def test_bin_accuracy(self):
        y_true = self.test_data.y_true.apply(np.argmax)
        y_pred = self.test_data.logits.apply(np.argmax)
        lens = self.test_data.lens

        self.assertEqual(self.accuracy.loc["Class A"][0],
                         _bin_accuracy(y_true,
                                       y_pred,
                                       lens,
                                       3,
                                       6))
        self.assertEqual(self.accuracy.loc["Class B"][0],
                         _bin_accuracy(y_true,
                                       y_pred,
                                       lens,
                                       0,
                                       3))



    def test_accuracy_vs_len(self):
        y_true = self.test_data.y_true.apply(np.argmax)
        y_pred = self.test_data.logits.apply(np.argmax)
        lens = self.test_data.lens
        bin_accuracy = pd.DataFrame(
                data={"accuracy": [.5, .75]},
                index=[4, 6]
                )
        acc = accuracy_vs_len(y_true, y_pred, lens, bins=2)
        res = all(np.isclose(acc.values, bin_accuracy.values)) \
                and list(acc.index) == list(bin_accuracy.index)
        self.assertTrue(res)

    def test_accuracy_vs_pos(self):
        pass

    def test_confusion_matrix(self):
        pass


if __name__ == "__main__":
    ut.main()

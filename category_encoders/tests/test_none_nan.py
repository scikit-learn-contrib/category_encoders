import unittest
import pandas as pd
import category_encoders as ce
import numpy as np

__author__ = 'willmcginnis'


class TestDist(unittest.TestCase):
    """
    """

    def test_dist(self):
        data = np.array([
            ['apple', None],
            ['peach', 'lemon']
        ])
        encoder = ce.OrdinalEncoder(impute_missing=True)
        encoder.fit(data)
        a = encoder.transform(data)
        print(a)
        self.assertEqual(a.values[0, 1], -1)
        self.assertEqual(a.values[1, 1], 0)

        encoder = ce.OrdinalEncoder(impute_missing=False)
        encoder.fit(data)
        a = encoder.transform(data)
        self.assertTrue(np.isnan(a.values[0, 1]))
        self.assertEqual(a.values[1, 1], 0)
import unittest
import pandas as pd
import category_encoders as ce
import numpy as np

__author__ = 'willmcginnis'


class TestDist(unittest.TestCase):
    """
    """

    def test_dist(self):
        data = np.array(['apple', 'orange', 'peach', 'lemon'])
        encoder = ce.BinaryEncoder()
        encoder.fit(data)

        # split dataframe into two transforms and recombine
        a = encoder.transform(data[:1])
        b = encoder.transform(data[1:])
        split = pd.concat([a, b])
        split = split.reset_index(drop=True)

        # run all at once
        c = encoder.transform(data)

        # make sure they are the same
        self.assertTrue(split.equals(c))
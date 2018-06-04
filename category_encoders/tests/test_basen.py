import category_encoders as ce
import unittest
import pandas as pd

__author__ = 'willmcginnis'


class TestBasen(unittest.TestCase):
    """
    """

    def test_basen(self):
        df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['d', 'e', 'f']})
        df_1 = pd.DataFrame({'col1': ['a', 'b', 'd'], 'col2': ['d', 'e', 'f']})
        enc = ce.BaseNEncoder(verbose=1)
        enc.fit(df)
        print(enc.transform(df_1))

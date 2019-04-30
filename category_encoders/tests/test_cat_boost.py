import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders


class TestBinaryEncoder(TestCase):

    def test_catBoost(self):
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A']})
        y = pd.Series([1, 0, 1, 0, 1])
        enc = encoders.CatBoostEncoder()
        obtained = enc.fit_transform(X, y)
        self.assertEqual(list(obtained['col1']), [0.6, 0.6, 0.6/2, 0.6, 1.6/2], 'The nominator is incremented by the prior. The denominator by 1.')

        X_t = pd.DataFrame({'col1': ['B', 'B', 'A']})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [1.6/3, 1.6/3, 2.6/3])

    def test_catBoost_missing(self):
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A', None, None, None]})
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        enc = encoders.CatBoostEncoder(handle_missing='value')
        obtained = enc.fit_transform(X, y)
        self.assertEqual(list(obtained['col1']), [0.5, 0.5, 0.5/2, 0.5, 1.5/2, 0.5, 0.5/2, 1.5/3], 'We treat None as another category.')

        X_t = pd.DataFrame({'col1': ['B', 'B', 'A', None]})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [1.5/3, 1.5/3, 2.5/3, 1.5/4])

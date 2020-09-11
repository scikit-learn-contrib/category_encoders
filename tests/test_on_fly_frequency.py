import pandas as pd
import numpy as np
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders


class TestOnFlyFrequencyEncoder(TestCase):

    def test_on_fly_frequency_encoder(self):
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'B', 'C', 'A', 'A']})
        y = pd.Series([1, 0, 1, 0, 1, 0, 0])
        enc = encoders.OnFlyFrequencyEncoder()
        obtained = enc.fit_transform(X, y)
        self.assertEqual(list(obtained['col1']), [1.0, 1.0/2, 2.0/3, 3.0/4, 1.0/5, 2.0/6, 3.0/7], 'The nominator is incremented by the prior. The denominator by 1.')

        # For testing set, use statistics calculated on all the training data.
        # See: CatBoost: unbiased boosting with categorical features, page 4.
        X_t = pd.DataFrame({'col1': ['B', 'B', 'A', 'C']})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [3.0/7, 3.0/7, 3.0/7, 1.0/7])

    def test_on_fly_frequency_encoder_missing_in_train(self):
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A', np.NaN, np.NaN, np.NaN]})
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        enc = encoders.OnFlyFrequencyEncoder(handle_missing='value')
        obtained = enc.fit_transform(X, y)
        self.assertEqual(list(obtained['col1']), [1.0, 1.0/2, 2.0/3, 1.0/4, 2.0/5, 1.0/6, 2.0/7, 3.0/8], 'We treat None as another category.')
        X_t = pd.DataFrame({'col1': ['B', 'B', 'C', np.NaN]})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [2.0/8, 2.0/8, 1.0/8, 3.0/8])

    def test_on_fly_frequency_encoder_missing_in_test(self):
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A']})
        y = pd.Series([1, 0, 1, 0, 1])
        enc = encoders.OnFlyFrequencyEncoder(handle_missing='value')
        obtained = enc.fit_transform(X, y)
        X_t = pd.DataFrame({'col1': ['B', 'B', 'C', np.NaN, 'D']})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [2.0/5, 2.0/5, 1.0/5, 1.0/4, 1.0/4])

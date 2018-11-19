import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.test_utils as tu
import numpy as np

import category_encoders as encoders

X = tu.create_dataset(n_rows=100)

class TestHasingEncoder(TestCase):

    def test_get_feature_names(self):
        enc = encoders.HashingEncoder()
        enc.fit(X)
        obtained = enc.get_feature_names()
        expected = 8 # length of feature name list
        self.assertEquals(len(obtained), expected)

    def test_get_feature_names_names_not_set(self):
        enc = encoders.HashingEncoder()
        self.assertRaises(ValueError, enc.get_feature_names)

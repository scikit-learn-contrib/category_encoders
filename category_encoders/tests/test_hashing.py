import pandas as pd
from unittest import TestCase  # or `from unittest2 import ...` if on Python < 3.4

import category_encoders as encoders

class TestHashingEncoder(TestCase):

    def test_get_feature_names(self):
        X = pd.DataFrame([['A','B','C']], columns=['col1', 'col2', 'col3'])
        enc = encoders.HashingEncoder(cols=['col2', 'col3'])
        result = enc.fit_transform(X)
        obtained = enc.get_feature_names()
        print(obtained)
        expected = set(result.columns) - set('col1')
        print(expected)

        self.assertEqual(len(expected), len(obtained))

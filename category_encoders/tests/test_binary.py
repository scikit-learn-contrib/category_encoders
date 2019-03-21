import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import random
from sklearn.compose import ColumnTransformer

import category_encoders as encoders


class TestBinaryEncoder(TestCase):

    def test_binary_bin(self):
        data = np.array(['a', 'ba', 'ba'])
        out = encoders.BinaryEncoder().fit_transform(data)
        self.assertTrue(pd.DataFrame([[0, 1], [1, 0], [1, 0]], columns=['0_0', '0_1']).equals(out))

    def test_binary_dist(self):
        data = np.array(['apple', 'orange', 'peach', 'lemon'])
        encoder = encoders.BinaryEncoder()
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

    def test_binary_return_missing_in_column_transformer(self):
        
        random.seed(2001)
        ds = [[random.choice(['A', 'B', 'C', np.nan])] for row in range(1000)]

        df = pd.DataFrame(ds, columns=['categorical_na'])
        df['categorical_na'] = pd.Categorical(df['categorical_na'], categories=['A', 'B', 'C'])
        
        X = df
        expected_na = X.loc[:,'categorical_na'].isna().sum()
        
        ct = ColumnTransformer([
            ("be",
             encoders.BinaryEncoder(handle_missing="return_nan"),
             ["categorical_na"])
        ])
        X_trans = ct.fit_transform(X)
        result_na = X.shape[0] - X_trans[~np.isnan(X_trans).all(axis=1)].shape[0]
        self.assertEqual(expected_na, result_na)

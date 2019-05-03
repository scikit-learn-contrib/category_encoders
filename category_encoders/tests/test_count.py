import random
import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.test_utils as tu
import numpy as np

import category_encoders as encoders


random.seed(1)

X = tu.create_dataset(n_rows=100, random_seed=0)
X_t = tu.create_dataset(n_rows=51, extras=True, random_seed=0)

class TestCountEncoder(TestCase):

    def test_count(self):

        enc = encoders.CountEncoder(verbose=1)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(0, set(out['extra'].values))
        self.assertIn(0.0, out.extra.unique())
        self.assertFalse(enc.mapping is None)
        self.assertTrue(len(enc.mapping) > 0)

    def test_count_min_group_size_int(self):
        # no group under threshold
        enc = encoders.CountEncoder(verbose=1, min_group_size=20)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)

        # single group under threshold
        enc = encoders.CountEncoder(verbose=1, min_group_size=30)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)

        # multiple groups under threshold
        enc = encoders.CountEncoder(verbose=1, min_group_size=35)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 3)

        # multiple groups under threshold with own name
        enc = encoders.CountEncoder(verbose=1, min_group_size=35, min_group_name='dave')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertTrue(enc.mapping['extra'].index.isin(['dave']).any())

    def test_count_min_group_size_float(self):
        # no group under threshold
        enc = encoders.CountEncoder(verbose=1, min_group_size=0.2)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)

        # single group under threshold
        enc = encoders.CountEncoder(verbose=1, min_group_size=0.3)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)

        # multiple groups under threshold
        enc = encoders.CountEncoder(verbose=1, min_group_size=0.35)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 3)

    def test_count_impute_missing(self):
        enc = encoders.CountEncoder(verbose=1, handle_unknown='ignore')
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(out.extra.isna().any())

    def test_count_combine_min_nan_groups(self):
        enc = encoders.CountEncoder(verbose=1, combine_min_nan_groups=False)
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertEqual(len(set(out['none'].values)), 3)

    def test_count_count_nan_fit(self):
        enc = encoders.CountEncoder(verbose=1, count_nan_fit=False)
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertEqual(len(set(out['none'].values)), 3)

    def test_count_normalize(self):
        enc = encoders.CountEncoder(verbose=1, count_nan_fit=False)
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(out['none'].dtype == float)
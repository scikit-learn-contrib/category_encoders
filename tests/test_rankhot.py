import pandas as pd
from unittest import TestCase
import tests.helpers as th
import numpy as np
import category_encoders as encoders


np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestRankHotEncoder(TestCase):

    def test_handleNaNvalue(self):
        enc = encoders.RankHotEncoder(handle_unknown='value', cols=['none'])
        enc.fit(X)
        t_f = enc.transform(X)
        inv_tf = enc.inverse_transform(t_f)
        self.assertEqual(t_f.shape[1]-(X.shape[1]-1), len(X.none.unique()))
        self.assertTupleEqual(inv_tf.shape,X.shape)

    def test_handleCategoricalValue(self):
        enc = encoders.RankHotEncoder(cols=['categorical'])
        enc.fit(X)
        t_f = enc.transform(X)
        inv_tf = enc.inverse_transform(t_f)
        self.assertEqual(t_f.shape[1] - (X.shape[1] - 1), len(X.categorical.unique()))
        self.assertTupleEqual(inv_tf.shape, X.shape)

    def test_naCatagoricalValue(self):
        enc = encoders.RankHotEncoder(handle_unknown='value', cols=['na_categorical'])
        enc.fit(X)
        t_f = enc.transform(X)
        inv_tf = enc.inverse_transform(t_f)
        self.assertTupleEqual(inv_tf.shape, X.shape)

    def test_extraValue(self):
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        enc = encoders.RankHotEncoder(handle_unknown='value')
        enc.fit(train)
        t_f = enc.transform(test)
        inv_tf = enc.inverse_transform(t_f)
        self.assertEqual(t_f.shape[1] - (train.shape[1] - 1), len(test.city.unique()), "All the extra values are displayed as None after inverse transform")
        self.assertTupleEqual(inv_tf.shape, train.shape)

    def test_invariant(self):
        enc = encoders.RankHotEncoder(cols=['invariant'], drop_invariant=True)
        enc.fit(X)
        self.assertNotEqual(X.shape[1], len(enc.feature_names_out_))

    def test_categoricalNaming(self):
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        enc = encoders.RankHotEncoder(use_cat_names=True)
        enc.fit(train)
        tf = enc.transform(train)
        self.assertListEqual(['city_chicago', 'city_st louis'], list(tf.columns))

    def test_rankhot(self):
        enc = encoders.RankHotEncoder(verbose=1)
        enc.fit(X)
        t_f = enc.transform(X)
        inv_tf = enc.inverse_transform(t_f)
        self.assertTupleEqual(X.shape, inv_tf.shape, "Check shape doesn't change")



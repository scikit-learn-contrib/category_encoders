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
        train = pd.DataFrame({'city': ['chicago', 'st louis', 'chicago', "st louis"]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        enc = encoders.RankHotEncoder(handle_unknown='value')
        train_out = enc.fit_transform(train)
        expected_mapping = pd.DataFrame([[1, 0],[1, 1],], columns=["city_1", "city_2"], index=[1,2])
        expected_out_train = pd.DataFrame([[1, 0],[1, 1],[1, 0],[1, 1],], columns=["city_1", "city_2"])
        expected_out_test = pd.DataFrame([[1, 0],[0, 0],], columns=["city_1", "city_2"])
        pd.testing.assert_frame_equal(train_out, expected_out_train)
        pd.testing.assert_frame_equal(enc.mapping[0]["mapping"], expected_mapping)
        t_f = enc.transform(test)
        pd.testing.assert_frame_equal(t_f, expected_out_test)
        inv_tf = enc.inverse_transform(t_f)
        expected_inverse_test = pd.DataFrame({'city': ['chicago', "None"]})
        th.verify_inverse_transform(expected_inverse_test, inv_tf)

    def test_invariant(self):
        enc = encoders.RankHotEncoder(cols=['invariant'], drop_invariant=True)
        enc.fit(X)
        self.assertFalse(any([c.startswith("invariant") for c in enc.feature_names_out_]))
        self.assertTrue(any([c.startswith("invariant") for c in enc.invariant_cols]))

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
        th.verify_inverse_transform(X, inv_tf)

    def test_order(self):
        """
        Since RankHotEncoding respects the order in ordinal variables, the mapping should be independent of input order
        """
        train_order_1 = pd.DataFrame({'grade': ['B', 'A', 'C', 'F', 'D', 'C', 'F', 'D'],
                                      "ord_var": [1, 3, 2, 2, 2, 1, 3, 1]})
        train_order_2 = pd.DataFrame({'grade': ['A', 'D', 'C', 'B', 'C', 'F', 'F', 'D'],
                                      "ord_var": [3, 1, 2, 2, 2, 1, 3, 1]})
        enc = encoders.RankHotEncoder(cols=["grade", "ord_var"])
        enc.fit(train_order_1)
        mapping_order_1 = enc.ordinal_encoder.mapping
        enc.fit(train_order_2)
        mapping_order_2 = enc.ordinal_encoder.mapping
        for m1, m2 in zip(mapping_order_1, mapping_order_2):
            self.assertEqual(m1["col"], m2["col"])
            pd.testing.assert_series_equal(m1["mapping"], m2["mapping"])


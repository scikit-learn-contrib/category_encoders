import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
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


class TestOrdinalEncoder(TestCase):

    def test_ordinal(self):

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(-1, set(out['extra'].values))
        self.assertFalse(enc.mapping is None)
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, mapping=enc.mapping, return_df=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(-1, set(out['extra'].values))
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
        enc.fit(X)
        out = enc.transform(X_t)
        out_cats = [x for x in set(out['extra'].values) if np.isfinite(x)]
        self.assertEqual(len(out_cats), 3)
        self.assertFalse(enc.mapping is None)

    def test_ordinal_dist(self):
        data = np.array([
            ['apple', 'lemon'],
            ['peach', None]
        ])
        encoder = encoders.OrdinalEncoder()
        result = encoder.fit_transform(data)
        self.assertEqual(2, len(result[0].unique()), "We expect two unique values in the column")
        self.assertEqual(2, len(result[1].unique()), "We expect two unique values in the column")
        self.assertFalse(np.isnan(result.values[1, 1]))

        encoder = encoders.OrdinalEncoder(handle_missing='return_nan')
        result = encoder.fit_transform(data)
        self.assertEqual(2, len(result[0].unique()), "We expect two unique values in the column")
        self.assertEqual(2, len(result[1].unique()), "We expect two unique values in the column")

    def test_pandas_categorical(self):
        X = pd.DataFrame({
            'Str': ['a', 'c', 'c', 'd'],
            'Categorical': pd.Categorical(list('bbea'), categories=['e', 'a', 'b'], ordered=True)
        })

        enc = encoders.OrdinalEncoder()
        out = enc.fit_transform(X)

        th.verify_numeric(out)
        self.assertEqual(3, out['Categorical'][0])
        self.assertEqual(3, out['Categorical'][1])
        self.assertEqual(1, out['Categorical'][2])
        self.assertEqual(2, out['Categorical'][3])

    def test_handle_missing_have_nan_fit_time_expect_as_category(self):
        train = pd.DataFrame({'city': ['chicago', np.nan],
                              'city_cat': pd.Categorical(['chicago', np.nan])})

        enc = encoders.OrdinalEncoder(handle_missing='value')
        out = enc.fit_transform(train)

        self.assertListEqual([1, 2], out['city'].tolist())
        self.assertListEqual([1, 2], out['city_cat'].tolist())

    def test_handle_missing_have_nan_transform_time_expect_negative_2(self):
        train = pd.DataFrame({'city': ['chicago', 'st louis'],
                              'city_cat': pd.Categorical(['chicago', 'st louis'])})
        test = pd.DataFrame({'city': ['chicago', np.nan],
                             'city_cat': pd.Categorical(['chicago', np.nan])})

        enc = encoders.OrdinalEncoder(handle_missing='value')
        enc.fit(train)
        out = enc.transform(test)

        self.assertListEqual([1, -2], out['city'].tolist())
        self.assertListEqual([1, -2], out['city_cat'].tolist())

    def test_handle_unknown_have_new_value_expect_negative_1(self):
        # See #238
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        expected = [1.0, -1.0]

        enc = encoders.OrdinalEncoder(handle_missing='return_nan')
        enc.fit(train)
        result = enc.transform(test)['city'].tolist()

        self.assertEqual(expected, result)

    def test_handle_unknown_have_new_value_expect_negative_1_categorical(self):
        cities = ['st louis', 'chicago', 'los angeles']
        train = pd.DataFrame({'city': pd.Categorical(cities[:-1], categories=cities)})
        test = pd.DataFrame({'city': pd.Categorical(cities[1:], categories=cities)})

        expected = [2.0, -1.0]

        enc = encoders.OrdinalEncoder(handle_missing='return_nan')
        enc.fit(train)
        result = enc.transform(test)['city'].tolist()

        self.assertEqual(expected, result)

    def test_custom_mapping(self):
        # See issue 193
        custom_mapping = [{'col': 'col1', 'mapping': {np.NaN: 0, 'a': 1, 'b': 2}},  # The mapping from the documentation
                          {'col': 'col2', 'mapping': {np.NaN: -3, 'x': 11, 'y': 2}}]

        train = pd.DataFrame({'col1': ['a', 'a', 'b', np.NaN],
                              'col2': ['x', 'y', np.NaN, np.NaN]})

        enc = encoders.OrdinalEncoder(handle_missing='value', mapping=custom_mapping)
        out = enc.fit_transform(train)  # We have to first 'fit' before 'transform' even if we do nothing during the fit...

        self.assertListEqual([1, 1, 2, 0], out['col1'].tolist())
        self.assertListEqual([11, 2, -3, -3], out['col2'].tolist())

    def test_HaveNegativeOneInTrain_ExpectCodedAsOne(self):
        train = pd.DataFrame({'city': [-1]})
        expected = [1]

        enc = encoders.OrdinalEncoder(cols=['city'])
        result = enc.fit_transform(train)['city'].tolist()

        self.assertEqual(expected, result)

    def test_HaveNaNInTrain_ExpectCodedAsOne(self):
        train = pd.DataFrame({'city': [np.nan]})
        expected = [1]

        enc = encoders.OrdinalEncoder(cols=['city'])
        result = enc.fit_transform(train)['city'].tolist()

        self.assertEqual(expected, result)

    def test_HaveNoneAndNan_ExpectCodesAsOne(self):
        train = pd.DataFrame({'city': [np.nan, None]})
        expected = [1, 1]

        enc = encoders.OrdinalEncoder(cols=['city'])
        result = enc.fit_transform(train)['city'].tolist()

        self.assertEqual(expected, result)

    def test_inverse_transform_HaveUnknown_ExpectWarning(self):
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='value')
        enc.fit(train)
        result = enc.transform(test)

        message = 'inverse_transform is not supported because transform impute '\
                             'the unknown category -1 when encode city'
        
        with self.assertWarns(UserWarning, msg=message) as w:
            enc.inverse_transform(result)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingValue_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingReturnNan_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OrdinalEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_BothFieldsAreReturnNanWithNan_ExpectValueError(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OrdinalEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        
        message = 'inverse_transform is not supported because transform impute '\
                             'the unknown category nan when encode city'

        with self.assertWarns(UserWarning, msg=message) as w:
            enc.inverse_transform(result)

    def test_inverse_transform_HaveMissingAndNoUnknown_ExpectInversed(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveHandleMissingValueAndHandleUnknownReturnNan_ExpectBestInverse(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})

        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_with_mapping(self):
        df = X.copy(deep=True)
        categoricals = ['unique_int', 'unique_str', 'invariant', 'underscore', 'none', 'extra', 321]
        mapping = [{'col': c, 'mapping': pd.Series(data=range(len(df[c].unique())), index=df[c].unique()), 'data_type': X[c].dtype} for c in categoricals]
        enc = encoders.OrdinalEncoder(cols=categoricals, handle_unknown='ignore', mapping=mapping, return_df=True)
        df[categoricals] = enc.fit_transform(df[categoricals])

        recovered = enc.inverse_transform(df[categoricals])

        pd.testing.assert_frame_equal(X[categoricals], recovered)

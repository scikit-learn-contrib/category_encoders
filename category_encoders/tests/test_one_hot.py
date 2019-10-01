import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import category_encoders.tests.helpers as th

import category_encoders as encoders


np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestOneHotEncoderTestCase(TestCase):

    def test_one_hot(self):
        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertEqual(enc.transform(X_t).shape[1],
                         enc.transform(X).shape[1],
                         'We have to get the same count of columns despite the presence of a new value')

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='indicator')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_-1', out.columns.values)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len([x for x in out.columns.values if str(x).startswith('extra_')]), 3)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='error')
        # The exception is already raised in fit() because transform() is called there to get
        # feature_names right.
        enc.fit(X)
        with self.assertRaises(ValueError):
            enc.transform(X_t)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='return_nan', use_cat_names=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_A', out.columns.values)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, use_cat_names=True, handle_unknown='indicator')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_-1', out.columns.values)

        # test inverse_transform
        X_i = th.create_dataset(n_rows=100, has_missing=False)
        X_i_t = th.create_dataset(n_rows=50, has_missing=False)
        cols = ['underscore', 'none', 'extra', 321, 'categorical']

        enc = encoders.OneHotEncoder(verbose=1, use_cat_names=True, cols=cols)
        enc.fit(X_i)
        obtained = enc.inverse_transform(enc.transform(X_i_t))
        th.verify_inverse_transform(X_i_t, obtained)

    def test_fit_transform_HaveMissingValuesAndUseCatNames_ExpectCorrectValue(self):
        encoder = encoders.OneHotEncoder(cols=[0], use_cat_names=True, handle_unknown='indicator', return_df=False)

        result = encoder.fit_transform([[-1]])

        self.assertListEqual([[1, 0]], result.tolist())

    def test_inverse_transform_HaveDedupedColumns_ExpectCorrectInverseTransform(self):
        encoder = encoders.OneHotEncoder(cols=['match', 'match_box'], use_cat_names=True)
        value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series(-1)})

        transformed = encoder.fit_transform(value)
        inverse_transformed = encoder.inverse_transform(transformed)

        assert value.equals(inverse_transformed)

    def test_inverse_transform_HaveNoCatNames_ExpectCorrectInverseTransform(self):
        encoder = encoders.OneHotEncoder(cols=['match', 'match_box'], use_cat_names=False)
        value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series(-1)})

        transformed = encoder.fit_transform(value)
        inverse_transformed = encoder.inverse_transform(transformed)

        assert value.equals(inverse_transformed)

    def test_fit_transform_HaveColumnAppearTwice_ExpectColumnsDeduped(self):
        encoder = encoders.OneHotEncoder(cols=['match', 'match_box'], use_cat_names=True, handle_unknown='indicator')
        value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series('-1')})

        result = encoder.fit_transform(value)
        columns = result.columns.tolist()

        self.assertSetEqual({'match_box_-1', 'match_-1', 'match_box_-1#', 'match_box_-1##'}, set(columns))

    def test_fit_transform_HaveHandleUnknownValueAndUnseenValues_ExpectAllZeroes(self):
        train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
        test = pd.DataFrame({'city': ['Chicago', 'Detroit']})
        expected_result = pd.DataFrame({'city_1': [1, 0],
                                        'city_2': [0, 0]},
                                       columns=['city_1', 'city_2'])

        enc = encoders.OneHotEncoder(handle_unknown='value')
        result = enc.fit(train).transform(test)

        pd.testing.assert_frame_equal(expected_result, result)

    def test_fit_transform_HaveHandleUnknownValueAndSeenValues_ExpectMappingUsed(self):
        train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
        expected_result = pd.DataFrame({'city_1': [1, 0],
                                        'city_2': [0, 1]},
                                       columns=['city_1', 'city_2'])

        enc = encoders.OneHotEncoder(handle_unknown='value')
        result = enc.fit(train).transform(train)

        pd.testing.assert_frame_equal(expected_result, result)

    def test_fit_transform_HaveHandleUnknownIndicatorAndNoMissingValue_ExpectExtraColumn(self):
        train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
        expected_result = pd.DataFrame({'city_1': [1, 0],
                                        'city_2': [0, 1],
                                        'city_-1': [0, 0]},
                                       columns=['city_1', 'city_2', 'city_-1'])

        enc = encoders.OneHotEncoder(handle_unknown='indicator')
        result = enc.fit(train).transform(train)

        pd.testing.assert_frame_equal(expected_result, result)

    def test_fit_transform_HaveHandleUnknownIndicatorAndMissingValue_ExpectValueSet(self):
        train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
        test = pd.DataFrame({'city': ['Chicago', 'Detroit']})
        expected_result = pd.DataFrame({'city_1': [1, 0],
                                        'city_2': [0, 0],
                                        'city_-1': [0, 1]},
                                       columns=['city_1', 'city_2', 'city_-1'])

        enc = encoders.OneHotEncoder(handle_unknown='indicator')
        result = enc.fit(train).transform(test)

        pd.testing.assert_frame_equal(expected_result, result)

    def test_HandleMissingIndicator_NanInTrain_ExpectAsColumn(self):
        train = ['A', 'B', np.nan]

        encoder = encoders.OneHotEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_HaveNoNan_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.OneHotEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [[1, 0, 0],
                    [0, 1, 0]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_NanNoNanInTrain_ExpectAsNanColumn(self):
        train = ['A', 'B']
        test = ['A', 'B', np.nan]

        encoder = encoders.OneHotEncoder(handle_missing='indicator', handle_unknown='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveNoUnknownInTrain_ExpectIndicatorInTest(self):
        train = ['A', 'B']
        test = ['A', 'B', 'C']

        encoder = encoders.OneHotEncoder(handle_unknown='indicator', handle_missing='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveOnlyKnown_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.OneHotEncoder(handle_unknown='indicator', handle_missing='value')
        result = encoder.fit_transform(train)

        expected = [[1, 0, 0],
                    [0, 1, 0]]
        self.assertEqual(result.values.tolist(), expected)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingValue_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OneHotEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingReturnNan_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OneHotEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_BothFieldsAreReturnNanWithNan_ExpectValueError(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OneHotEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        
        message = 'inverse_transform is not supported because transform impute '\
                  'the unknown category nan when encode city'

        with self.assertWarns(UserWarning, msg=message) as w:
            enc.inverse_transform(result)

    def test_inverse_transform_HaveMissingAndNoUnknown_ExpectInversed(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OneHotEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveHandleMissingValueAndHandleUnknownReturnNan_ExpectBestInverse(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})

        enc = encoders.OneHotEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(expected, original)

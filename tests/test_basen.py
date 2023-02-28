import pandas as pd
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import category_encoders as encoders
from .helpers import list_to_dataframe

class TestBaseNEncoder(TestCase):

    def test_fit_transform_have_base_2_expect_Correct_Encoding(self):
        train = pd.Series(['a', 'b', 'c', 'd'])

        result = encoders.BaseNEncoder(base=2).fit_transform(train)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_inverse_transform_HaveData_ExpectResultReturned(self):
        train = pd.Series(list('abcd')).to_frame('letter')

        enc = encoders.BaseNEncoder(base=2)
        result = enc.fit_transform(train)
        inversed_result = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, inversed_result)

    def test_HaveIndicatorAndNanValue_ExpectNewColumn(self):
        train = pd.Series(['a', 'b', 'c', np.nan])

        result = encoders.BaseNEncoder(handle_missing='indicator', base=2).fit_transform(train)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_HandleMissingIndicator_HaveNoNan_ExpectThirdColumn(self):
        train = pd.Series(['a', 'b', 'c'])

        result = encoders.BaseNEncoder(handle_missing='indicator', base=2).fit_transform(train)

        self.assertEqual(3, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())

    def test_HandleMissingIndicator_NanNoNanInTrain_ExpectAsNanColumn(self):
        train = pd.Series(['a', 'b', 'c'])
        test = pd.Series(['a', 'b', 'c', np.nan])

        encoder = encoders.BaseNEncoder(handle_missing='indicator')
        encoder.fit(train)
        result = encoder.transform(test)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_HandleUnknown_HaveUnknown_ExpectIndicatorInTest(self):
        train = ['A', 'B', 'C']
        test = ['A', 'B', 'C', 'D']
        train = list_to_dataframe(train)
        test = list_to_dataframe(test)

        encoder = encoders.BaseNEncoder(handle_unknown='indicator')
        encoder.fit(train)
        result = encoder.transform(test)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_HandleUnknown_HaveOnlyKnown_ExpectSecondColumn(self):
        train = ['A', 'B']
        train = list_to_dataframe(train)

        encoder = encoders.BaseNEncoder(handle_unknown='indicator')
        result = encoder.fit_transform(train)

        self.assertEqual(2, result.shape[0])
        self.assertListEqual([0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([1, 0], result.iloc[1, :].tolist())

    def test_inverse_transform_HaveNanInTrainAndHandleMissingValue_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.BaseNEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingReturnNan_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.BaseNEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_BothFieldsAreReturnNanWithNan_ExpectValueError(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.BaseNEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        
        message = 'inverse_transform is not supported because transform impute '\
                  'the unknown category nan when encode city'

        with self.assertWarns(UserWarning, msg=message) as w:
            enc.inverse_transform(result)

    def test_inverse_transform_HaveMissingAndNoUnknown_ExpectInversed(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.BaseNEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveHandleMissingValueAndHandleUnknownReturnNan_ExpectBestInverse(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})

        enc = encoders.BaseNEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_HaveRegexMetacharactersInColumnName_ExpectInversed(self):
        train = pd.DataFrame({'state (2-letter code)': ['il', 'ny', 'ca']})

        enc = encoders.BaseNEncoder()
        enc.fit(train)
        result = enc.transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_num_cols(self):
        """
        Test that BaseNEncoder produces the correct number of output columns.

        Since the value 0 is reserved for encoding unseen values, there need to be enough digits to
        represent up to nvals + 1 distinct encodings, where nvals is the number of distinct input
        values. This is ceil(log(nvals + 1, base)) digits.

        This test specifically checks the case where BaseNEncoder is initialized with
        handle_unknown='value' and handle_missing='value' (i.e. the defaults).
        """
        def num_cols(nvals, base):
            """Returns the number of columns output for a given number of distinct input values"""
            vals = [str(i) for i in range(nvals)]
            df = pd.DataFrame({'vals': vals})
            encoder = encoders.BaseNEncoder(base=base)
            encoder.fit(df)
            return len(list(encoder.transform(df)))

        self.assertEqual(num_cols(1, 2), 1)
        self.assertEqual(num_cols(2, 2), 2)
        self.assertEqual(num_cols(3, 2), 2)
        self.assertEqual(num_cols(4, 2), 3)
        self.assertEqual(num_cols(7, 2), 3)
        self.assertEqual(num_cols(8, 2), 4)
        self.assertEqual(num_cols(62, 2), 6)
        self.assertEqual(num_cols(63, 2), 6)
        self.assertEqual(num_cols(64, 2), 7)
        self.assertEqual(num_cols(65, 2), 7)

        # nvals = 0 returns the original dataframe unchanged, so it still has 1 column even though
        # logically there should be zero.
        self.assertEqual(num_cols(0, 2), 1)

        self.assertEqual(num_cols(55, 7), 3)

"""Tests for the BaseNEncoder class."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd


class TestBaseNEncoder(TestCase):
    """Tests for the BaseNEncoder class."""

    def test_fit_transform_have_base_2_expect_correct_encoding(self):
        """Test the BaseNEncoder with base 2."""
        train = pd.Series(['a', 'b', 'c', 'd'])

        result = encoders.BaseNEncoder(base=2).fit_transform(train)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_inverse_transform(self):
        """Test the BaseNEncoder inverse_transform method."""
        train = pd.Series(list('abcd')).to_frame('letter')

        enc = encoders.BaseNEncoder(base=2)
        result = enc.fit_transform(train)
        inversed_result = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, inversed_result)

    def test_handle_missing_indicator_with_nan(self):
        """Test the BaseNEncoder with handle_missing='indicator'."""
        train = pd.Series(['a', 'b', 'c', np.nan])

        result = encoders.BaseNEncoder(handle_missing='indicator', base=2).fit_transform(train)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_handle_missing_indicator_without_nan(self):
        """Test the BaseNEncoder with handle_missing='indicator'.

        This should add a column for predict if there was no missing value in the training set.
        """
        train = pd.Series(['a', 'b', 'c'])

        encoder = encoders.BaseNEncoder(handle_missing='indicator', base=2)
        result = encoder.fit_transform(train)

        self.assertEqual(3, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())

        with self.subTest("should work with a missing value in test set"):
            test = pd.Series(['a', 'b', 'c', np.nan])

            result = encoder.transform(test)

            self.assertEqual(4, result.shape[0])
            self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
            self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
            self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
            self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_handle_unknown_indicator(self):
        """Test the BaseNEncoder with handle_unknown='indicator'."""
        train = ['A', 'B', 'C']
        test = ['A', 'B', 'C', 'D']

        encoder = encoders.BaseNEncoder(handle_unknown='indicator')
        encoder.fit(train)
        result = encoder.transform(test)

        self.assertEqual(4, result.shape[0])
        self.assertListEqual([0, 0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([0, 1, 0], result.iloc[1, :].tolist())
        self.assertListEqual([0, 1, 1], result.iloc[2, :].tolist())
        self.assertListEqual([1, 0, 0], result.iloc[3, :].tolist())

    def test_handle_unknown_indicator_no_unknowns(self):
        """Should create an indicator column even if no unknown values appear in the test set."""
        train = ['A', 'B']

        encoder = encoders.BaseNEncoder(handle_unknown='indicator')
        result = encoder.fit_transform(train)

        self.assertEqual(2, result.shape[0])
        self.assertListEqual([0, 1], result.iloc[0, :].tolist())
        self.assertListEqual([1, 0], result.iloc[1, :].tolist())

    def test_inverse_transform_have_nan_in_train(self):
        """Test the BaseNEncoder inverse_transform method with NaN in the training set."""
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        handle_missing = ["value", "return_nan"]
        for handle_missing_strategy in handle_missing:
            with self.subTest(f"Should work for handle_missing='{handle_missing_strategy}"):
                enc = encoders.BaseNEncoder(handle_missing=handle_missing_strategy,
                                            handle_unknown='value')
                result = enc.fit_transform(train)
                original = enc.inverse_transform(result)
                pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_not_supported_with_unknown_values(self):
        """Test that inverse_transform is not supported if a nan could be either missing or unknown.

        This happens if both handle_missing and handle_unkown are set to 'return_nan'.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.BaseNEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)

        message = (
            'inverse_transform is not supported because transform impute '
            'the unknown category nan when encode city'
        )

        with self.assertWarns(UserWarning, msg=message):
            enc.inverse_transform(result)

    def test_inverse_transform_with_missing_and_unknown(self):
        """Test the BaseNEncoder inverse_transform method with missing and unknown values.

        In the case of handle_missing='value' and handle_unknown='return_nan',
        the inverse_transform can distinguish between missing and unknown values and
        hence should work. Unknown values are encoded as nan in the inverse.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.BaseNEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        with self.subTest("should work with only unknown values"):
            test = pd.DataFrame({'city': ['chicago', 'los angeles']})
            result = enc.transform(test)
            original = enc.inverse_transform(result)
            pd.testing.assert_frame_equal(train, original)

        with self.subTest("should inverse transform unknowns and missing values to NaN"):
            test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
            expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})
            result = enc.transform(test)
            original = enc.inverse_transform(result)
            pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_have_regex_metacharacters_in_column_name(self):
        """Test the inverse_transform method with regex metacharacters in column name."""
        train = pd.DataFrame({'state (2-letter code)': ['il', 'ny', 'ca']})

        enc = encoders.BaseNEncoder()
        enc.fit(train)
        result = enc.transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_num_cols(self):
        """Test that BaseNEncoder produces the correct number of output columns.

        Since the value 0 is reserved for encoding unseen values, there need to be enough digits to
        represent up to nvals + 1 distinct encodings, where nvals is the number of distinct input
        values. This is ceil(log(nvals + 1, base)) digits.

        This test specifically checks the case where BaseNEncoder is initialized with
        handle_unknown='value' and handle_missing='value' (i.e. the defaults).
        """

        def num_cols(nvals, base):
            """Returns the number of columns output for a given number of distinct input values."""
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

"""Tests for the BaseNEncoder class."""
import warnings
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest


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

        This happens if both handle_missing and handle_unknown are set to 'return_nan'.
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


class TestBaseNMutationHardening(TestCase):
    """Targeted tests for survivors of the mutation-testing campaign in basen."""

    def test_base1_round_trip_and_no_input_mutation(self):
        """Base-1 encoding is an invertible one-hot that must not mutate its input.

        Kills the base-1 branch of basen_to_integer (mutmut_18/-19/-20/-21/-24) and pins
        the round-trip contract exercised by inverse_transform.
        """
        df = pd.DataFrame({'x': ['a', 'b', 'c'], 'num': [1, 2, 3]})
        snapshot = df.copy()
        encoder = encoders.BaseNEncoder(base=1)
        encoded = encoder.fit_transform(df)
        self.assertEqual(4, len([col for col in encoded.columns if col.startswith('x_')]))
        restored = encoder.inverse_transform(encoded)
        pd.testing.assert_frame_equal(df, restored)
        pd.testing.assert_frame_equal(snapshot, df)

    def test_inverse_transform_works_with_drop_invariant_and_no_invariant_columns(self):
        """inverse_transform works with drop_invariant=True when nothing was dropped.

        Kills basen.xǁBaseNEncoderǁinverse_transform__mutmut_14 (and→or in the guard).
        """
        df = pd.DataFrame({'x': ['a', 'b', 'c']})
        encoder = encoders.BaseNEncoder(base=2, drop_invariant=True)
        encoder.fit(df)
        restored = encoder.inverse_transform(encoder.transform(df))
        pd.testing.assert_frame_equal(df, restored)

    def test_inverse_transform_does_not_mutate_encoded_input(self):
        """inverse_transform must not modify the encoded frame it receives.

        Kills the deep-copy mutants of basen.xǁBaseNEncoderǁinverse_transform
        (mutmut_9/-12/-13).
        """
        df = pd.DataFrame({'x': ['a', 'b', 'c', 'd']})
        encoder = encoders.BaseNEncoder(base=2)
        encoder.fit(df)
        encoded = encoder.transform(df)
        snapshot = encoded.copy()
        encoder.inverse_transform(encoded)
        pd.testing.assert_frame_equal(snapshot, encoded)

    def test_inverse_transform_mixed_config_emits_no_inverse_warning(self):
        """handle_missing='return_nan' with handle_unknown='value' must not warn.

        Kills basen.xǁBaseNEncoderǁinverse_transform__mutmut_54 (and→or in the
        both-return_nan warning guard).
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        encoder = encoders.BaseNEncoder(handle_missing='return_nan', handle_unknown='value')
        encoder.fit(train)
        result = encoder.transform(train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            restored = encoder.inverse_transform(result)
        self.assertEqual(
            [], [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        )
        pd.testing.assert_frame_equal(train, restored)

    def test_return_df_false_round_trip_equality(self):
        """return_df=False survives transform and inverse_transform as ndarrays.

        Pins the ``X.to_numpy()`` return paths of both _transform and
        inverse_transform, and the round-trip equality of the encoded values.
        """
        df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'num': [1.5, 2.5, 3.5, 4.5]})
        encoder = encoders.BaseNEncoder(base=2, return_df=False)
        encoded = encoder.fit_transform(df)
        self.assertIsInstance(encoded, np.ndarray)
        restored = encoder.inverse_transform(encoded)
        self.assertIsInstance(restored, np.ndarray)
        converted = pd.DataFrame(restored, columns=df.columns)
        pd.testing.assert_frame_equal(df, converted, check_dtype=False)

    def test_base1_digit_calculation_skips_ceillogint(self):
        """base=1 reserves one digit per category instead of calling _ceillogint.

        _ceillogint rejects base < 2, so the base-1 branch of
        calc_required_digits is the only thing keeping base=1 encodings alive.
        """
        encoder = encoders.BaseNEncoder(base=1)
        self.assertEqual(4, encoder.calc_required_digits(['a', 'b', 'c']))

    def test_inverse_transform_error_contract(self):
        """inverse_transform raises ValueError when unfitted and on wrong-shape input."""
        df = pd.DataFrame({'x': ['a', 'b', 'c'], 'num': [1.5, 2.5, 3.5]})
        encoder = encoders.BaseNEncoder(base=2)
        with self.assertRaises(ValueError) as cm:
            encoder.inverse_transform(df)
        self.assertIn('Must train encoder', str(cm.exception))

        encoder.fit(df)
        # Encoded columns for x without the second column: basen_to_integer
        # succeeds, then the restored frame no longer matches the fitted width.
        encoded = encoder.transform(df)
        wrong_shape = encoded.drop(columns=['num'])
        with self.assertRaises(ValueError) as cm:
            encoder.inverse_transform(wrong_shape)
        self.assertIn('Unexpected input dimension', str(cm.exception))

    def test_ceillogint_invalid_base_error_path(self):
        """_ceillogint rejects base < 2 and n < 1 with ValueError."""
        from category_encoders.basen import _ceillogint

        with self.assertRaisesRegex(ValueError, 'base must be >= 2'):
            _ceillogint(5, 1)
        with self.assertRaisesRegex(ValueError, 'base must be >= 2'):
            _ceillogint(5, 0)
        with self.assertRaisesRegex(ValueError, 'n must be >= 1'):
            _ceillogint(0, 2)


def test_ceillogint_edge_cases_and_error_messages():
    """Exact integer-math outputs around the ceil boundary, plus both error contracts."""
    from category_encoders.basen import _ceillogint

    assert _ceillogint(1, 2) == 0
    assert _ceillogint(2, 2) == 1
    assert _ceillogint(3, 2) == 2
    assert _ceillogint(4, 2) == 2
    assert _ceillogint(5, 2) == 3
    assert _ceillogint(9, 3) == 2
    assert _ceillogint(10, 3) == 3
    with pytest.raises(ValueError, match=r'^base must be >= 2$'):
        _ceillogint(5, 1)
    with pytest.raises(ValueError, match=r'^base must be >= 2$'):
        _ceillogint(5, 0)
    with pytest.raises(ValueError, match=r'^n must be >= 1$'):
        _ceillogint(0, 2)
    with pytest.raises(ValueError, match=r'^n must be >= 1$'):
        _ceillogint(-3, 2)


def test_inverse_transform_before_fit_exact_message():
    enc = encoders.BaseNEncoder(cols=['c'])
    with pytest.raises(
        ValueError,
        match=r'^Must train encoder before it can be used to inverse_transform data$',
    ):
        enc.inverse_transform(pd.DataFrame({'c': [1]}))


def test_inverse_transform_wrong_dimension_exact_message():
    X = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['x', 'y', 'x']})
    enc = encoders.BaseNEncoder(cols=['col1'], base=2).fit(X)
    good = enc.transform(X)
    extra = good.copy()
    extra['junk'] = 1
    with pytest.raises(
        ValueError,
        match=rf'^Unexpected input dimension \d+, expected {enc._dim}$',
    ):
        enc.inverse_transform(extra)


def test_inverse_transform_drop_invariant_exact_message():
    X = pd.DataFrame({'col1': ['a', 'b', 'c'], 'const': ['x', 'x', 'x']})
    enc = encoders.BaseNEncoder(cols=['col1', 'const'], base=2, drop_invariant=True).fit(X)
    good = enc.transform(X)
    with pytest.raises(
        ValueError,
        match=r'^Unexpected input dimension \d+, the attribute drop_invariant should '
        r'be False when calling inverse_transform$',
    ):
        enc.inverse_transform(good)


def test_inverse_transform_return_nan_warning_full_message():
    train = pd.DataFrame({'city': ['chicago', 'st louis', np.nan]})
    enc = encoders.BaseNEncoder(
        cols=['city'], handle_unknown='return_nan', handle_missing='return_nan'
    )
    result = enc.fit_transform(train)
    with pytest.warns(UserWarning) as record:
        enc.inverse_transform(result)
    messages = [str(w.message) for w in record]
    assert messages == [
        'inverse_transform is not supported because transform impute '
        'the unknown category nan when encode city'
    ]


def test_basen_encode_golden_values_base3():
    """Four categories plus the sentinel need two base-3 digits; pins exact digit values."""
    X = pd.DataFrame({'x': ['a', 'b', 'c', 'd']})
    enc = encoders.BaseNEncoder(cols=['x'], base=3)
    out = enc.fit_transform(X)
    expected = pd.DataFrame({'x_0': [0, 0, 1, 1], 'x_1': [1, 2, 0, 1]})
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected, check_dtype=False)

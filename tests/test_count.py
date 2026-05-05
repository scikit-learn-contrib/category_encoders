"""Tests for the CountEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd

X = pd.DataFrame(
    {
        'none': [
            'A',
            'A',
            'B',
            None,
            None,
            'C',
            None,
            'C',
            None,
            'B',
            'A',
            'A',
            'C',
            'B',
            'B',
            'A',
            'A',
            None,
            'B',
            None,
        ],
        'na_categorical': [
            'A',
            'A',
            'C',
            'A',
            'B',
            'C',
            'C',
            'A',
            np.nan,
            'B',
            'A',
            'C',
            'C',
            'A',
            'B',
            'C',
            np.nan,
            'A',
            np.nan,
            np.nan,
        ],
    }
)

X_t = pd.DataFrame(
    {
        'none': ['A', 'C', None, 'B', 'C', 'C', None, None, 'A', 'A', 'C', 'A', 'B', 'A', 'A'],
        'na_categorical': [
            'C',
            'C',
            'A',
            'B',
            'C',
            'A',
            np.nan,
            'B',
            'A',
            'A',
            'B',
            np.nan,
            'A',
            np.nan,
            'A',
        ],
    }
)


class TestCountEncoder(TestCase):
    """Unit tests for the CountEncoder."""

    def test_count_defaults(self):
        """Test the defaults are working as expected on 'none' and 'categorical'.

        These are the most extreme edge cases for the count encoder.
        """
        enc = encoders.CountEncoder(verbose=1)
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(pd.Series([5, 3, 6]).isin(out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(pd.Series([6, 3]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(enc.mapping is not None)

    def test_count_handle_missing_string(self):
        """Test the handle_missing string on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(handle_missing='return_nan')

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_missing)
        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 3)
        self.assertTrue(pd.Series([6, 7, 3]).isin(out['na_categorical']).all())
        self.assertFalse(pd.Series([4]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isna().sum() == 3)

    def test_count_handle_missing_dict(self):
        """Test the handle_missing dict on 'none' and 'na_categorical'.

        We want to see differing behaviour between 'none' and 'na_cat' cols.
        """
        enc = encoders.CountEncoder(handle_missing={'na_categorical': 'return_nan'})

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_missing)
        self.assertTrue(pd.Series([5, 3, 6]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(pd.Series([6, 7, 3]).isin(out['na_categorical']).all())
        self.assertFalse(pd.Series([4]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isna().sum() == 3)

    def test_count_handle_unknown_string(self):
        """Test the handle_unknown string  on 'none' and 'na_categorical'.

        The 'handle_missing' must be set to 'return_nan' in order to test
        'handle_unknown' correctly.
        """
        enc = encoders.CountEncoder(
            handle_missing='return_nan',
            handle_unknown='return_nan',
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_unknown)
        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 3)
        self.assertTrue(pd.Series([3, 6, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isna().sum() == 3)

    def test_count_handle_unknown_dict(self):
        """Test the 'handle_unknown' dict with all non-default options."""
        enc = encoders.CountEncoder(
            handle_missing='return_nan',
            handle_unknown={'none': -1, 'na_categorical': 'return_nan'},
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_unknown)
        self.assertTrue(pd.Series([6, 5, 3, -1]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(pd.Series([3, 6, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isna().sum() == 3)

    def test_count_min_group_size_int(self):
        """Test the min_group_size int  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size=7)

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertIn(np.nan, enc.mapping['none'])
        self.assertTrue(pd.Series([13, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertIn('B_C_nan', enc.mapping['na_categorical'])
        self.assertFalse(np.nan in enc.mapping['na_categorical'])

    def test_count_min_group_size_dict(self):
        """Test the min_group_size dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size={'none': 6, 'na_categorical': 7})

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('none', enc._min_group_size)
        self.assertTrue(pd.Series([6, 8]).isin(out['none']).all())
        self.assertEqual(out['none'].unique().shape[0], 2)
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertIn(np.nan, enc.mapping['none'])
        self.assertTrue(pd.Series([13, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertIn('B_C_nan', enc.mapping['na_categorical'])
        self.assertFalse(np.nan in enc.mapping['na_categorical'])

    def test_count_combine_min_nan_groups_bool(self):
        """Test the min_nan_groups_bool on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size=7, combine_min_nan_groups=False)

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertEqual(out['none'].unique().shape[0], 3)
        self.assertEqual(out['none'].isna().sum(), 0)
        self.assertTrue(pd.Series([9, 7, 4]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 3)
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_combine_min_nan_groups_dict(self):
        """Test the combine_min_nan_groups dict  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={'none': 6, 'na_categorical': 7},
            combine_min_nan_groups={'none': 'force', 'na_categorical': False},
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._combine_min_nan_groups)
        self.assertTrue(pd.Series([14, 6]).isin(out['none']).all())
        self.assertEqual(out['none'].unique().shape[0], 2)
        self.assertEqual(out['none'].isna().sum(), 0)
        self.assertTrue(pd.Series([9, 7, 4]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 3)
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_min_group_name_string(self):
        """Test the min_group_name string on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size=6, min_group_name='dave')

        enc.fit(X)

        self.assertIn('dave', enc.mapping['none'])
        self.assertEqual(enc.mapping['none']['dave'], 8)
        self.assertIn('dave', enc.mapping['na_categorical'])
        self.assertEqual(enc.mapping['na_categorical']['dave'], 7)

    def test_count_min_group_name_dict(self):
        """Test the min_group_name dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={'none': 6, 'na_categorical': 6},
            min_group_name={'none': 'dave', 'na_categorical': None},
        )

        enc.fit(X)

        self.assertIn('none', enc._min_group_name)
        self.assertIn('dave', enc.mapping['none'])
        self.assertEqual(enc.mapping['none']['dave'], 8)
        self.assertIn('B_nan', enc.mapping['na_categorical'])
        self.assertEqual(enc.mapping['na_categorical']['B_nan'], 7)

    def test_count_normalize_bool(self):
        """Test the normalize bool on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size=6, normalize=True)

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._normalize)
        self.assertTrue(out['none'].round(5).isin([0.3, 0.4]).all())
        self.assertEqual(out['none'].unique().shape[0], 2)
        self.assertEqual(out['none'].isna().sum(), 0)
        self.assertTrue(pd.Series([0.3, 0.35]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 2)
        self.assertTrue(enc.mapping is not None)

    def test_count_normalize_dict(self):
        """Test the normalize dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=7, normalize={'none': True, 'na_categorical': False}
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._normalize)
        self.assertTrue(out['none'].round(5).isin([0.3, 0.15, 0.25]).all())
        self.assertEqual(out['none'].unique().shape[0], 3)
        self.assertEqual(out['none'].isna().sum(), 0)
        self.assertTrue(pd.Series([13, 7]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 2)
        self.assertTrue(enc.mapping is not None)

    def test_normalize_with_drop_invariant(self):
        """Test that normalize=True with drop_invariant=True does not incorrectly drop columns.

        Regression test for https://github.com/scikit-learn-contrib/category_encoders/issues/457.
        When normalize=True, encoded values are proportions with inherently small variance.
        The invariance check should not falsely flag these columns as invariant.
        """
        np.random.seed(42)
        n = 1000
        categories = [f"cat_{i}" for i in range(40)]
        data = np.random.choice(categories, size=n)
        df = pd.DataFrame({"col": data})

        enc = encoders.CountEncoder(
            drop_invariant=True,
            normalize=True,
            min_group_size=3,
            combine_min_nan_groups=True,
        )
        result = enc.fit_transform(df[['col']])

        # Column must not be dropped — it has many distinct proportions
        self.assertIn('col', result.columns)
        self.assertEqual(result.shape[0], n)
        self.assertGreater(result['col'].nunique(), 1)

    def test_normalize_with_float_min_group_size_and_drop_invariant(self):
        """Test normalize=True with float min_group_size and drop_invariant=True.

        Regression test for https://github.com/scikit-learn-contrib/category_encoders/issues/457.
        This is the exact combination reported in the issue.
        """
        np.random.seed(42)
        n = 1000
        categories = [f"cat_{i}" for i in range(40)]
        data = np.random.choice(categories, size=n)
        df = pd.DataFrame({"col": data})

        enc = encoders.CountEncoder(
            drop_invariant=True,
            normalize=True,
            min_group_size=0.03,
            combine_min_nan_groups=True,
        )
        result = enc.fit_transform(df[['col']])

        self.assertIn('col', result.columns)
        self.assertEqual(result.shape[0], n)
        self.assertGreater(result['col'].nunique(), 1)

    def test_min_group_size_float_with_normalize_false(self):
        """Test that float min_group_size is properly converted to int when normalize=False.

        Regression test for the dict truthiness bug on line 218 of count.py:
        'not self._normalize' (always False for non-empty dict) should be
        'not self._normalize[col]'.
        """
        # 20 rows, 4 categories: A=6, B=5, C=5, nan=4
        # min_group_size=0.30 means 30% of 20 = 6 rows threshold
        # With normalize=False, categories with count < 6 should be combined
        enc = encoders.CountEncoder(
            normalize=False,
            min_group_size=0.30,
            combine_min_nan_groups=True,
        )
        enc.fit(X[['none']])

        # The float 0.30 should have been converted to int (0.30 * 20 = 6)
        self.assertEqual(enc._min_group_size['none'], 0.30 * X.shape[0])

        # Categories below threshold (count < 6) should be combined
        self.assertTrue(len(enc._min_group_categories) > 0)

    def test_min_group_size_float_with_normalize_false_per_column(self):
        """Test float min_group_size conversion works per-column with mixed normalize dict."""
        enc = encoders.CountEncoder(
            normalize={'none': False, 'na_categorical': True},
            min_group_size=0.30,
        )
        enc.fit(X)

        # For 'none' (normalize=False): float should be converted to int count
        self.assertEqual(enc._min_group_size['none'], 0.30 * X.shape[0])

        # For 'na_categorical' (normalize=True): float should stay as-is (already a proportion)
        self.assertEqual(enc._min_group_size['na_categorical'], 0.30)

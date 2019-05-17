import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np
import category_encoders as encoders


X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=51, extras=True)

class TestCountEncoder(TestCase):

    def test_count_defaults(self):
        """Test the defaults are working as expected on 'extra', 'none' and 'categorical'"""
        enc = encoders.CountEncoder(verbose=1)
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(np.isin([32., 38., 30.], out['extra'].unique()).all())
        self.assertTrue(out['extra'].unique().shape == (4,))
        self.assertTrue(np.isin([31, 28, 21, 20], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([28, 25, 19], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(enc.mapping is not None)

    def test_count_handle_missing_string(self):
        """Test the handle_missing string on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            handle_missing='return_nan'
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_missing)
        self.assertTrue(np.isin([31, 28, 20], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 9)
        self.assertTrue(np.isin([28, 25], out['na_categorical'].unique()).all())
        self.assertFalse(np.isin([19], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(out['na_categorical'].isna().sum() == 13)

    def test_count_handle_missing_dict(self):
        """Test the handle_missing dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            handle_missing={'na_categorical': 'return_nan'}
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_missing)
        self.assertTrue(np.isin([31, 28, 21, 20], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([28, 25], out['na_categorical'].unique()).all())
        self.assertFalse(np.isin([19], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(out['na_categorical'].isna().sum() == 13)

    def test_count_handle_unknown_string(self):
        """Test the handle_unknown string  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            handle_missing='return_nan',
            handle_unknown='return_nan',
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_unknown)
        self.assertTrue(np.isin([31, 28, 20], out['none'].unique()).all())
        self.assertFalse(np.isin([21], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 9)
        self.assertTrue(np.isin([28, 25], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(out['na_categorical'].isna().sum() == 13)

    def test_count_handle_unknown_dict(self):
        """Test the """
        enc = encoders.CountEncoder(
            handle_missing='return_nan',
            handle_unknown={
                'none': -1,
                'na_categorical': 'return_nan'
            },
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_unknown)
        self.assertTrue(np.isin([31, 28, 20, -1], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([28, 25], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(out['na_categorical'].isna().sum() == 13)

    def test_count_min_group_size_int(self):
        """Test the min_group_size int  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size=25)

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertTrue(np.isin([31, 28, 41], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertIn('B_nan', enc.mapping['none'])
        self.assertTrue(np.isin([28, 25, 19], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_min_group_size_dict(self):
        """Test the min_group_size dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={'none': 25, 'na_categorical': 26}
        )

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('none', enc._min_group_size)
        self.assertTrue(np.isin([31, 28, 41], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertIn('B_nan', enc.mapping['none'])
        self.assertTrue(np.isin([28, 44], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertTrue(enc.mapping is not None)
        self.assertIn('A_nan', enc.mapping['na_categorical'])

    def test_count_combine_min_nan_groups_bool(self):
        """Test the min_nan_groups_bool on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=29,
            combine_min_nan_groups=False
        )

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertTrue(np.isin([31, 48, 21], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([28, 25, 19], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_combine_min_nan_groups_dict(self):
        """Test the combine_min_nan_groups dict  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={
                'none': 21,
                'na_categorical': 26
            },
            combine_min_nan_groups={
                'none': 'force',
                'na_categorical': False
            }
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._combine_min_nan_groups)
        self.assertTrue(np.isin([31, 41, 28], out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([28, 25, 19], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (3,))
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_min_group_name_string(self):
        """Test the min_group_name string on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=28,
            min_group_name='dave'
        )

        enc.fit(X)

        self.assertIn('dave', enc.mapping['none'])
        self.assertTrue(enc.mapping['none']['dave'] == 41)
        self.assertIn('dave', enc.mapping['na_categorical'])
        self.assertTrue(enc.mapping['na_categorical']['dave'] == 44)

    def test_count_min_group_name_dict(self):
        """Test the min_group_name dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={
                'none': 25, 'na_categorical': 26
            },
            min_group_name={
                'none': 'dave', 'na_categorical': None
            }
        )

        enc.fit(X)

        self.assertIn('none', enc._min_group_name)
        self.assertIn('dave', enc.mapping['none'])
        self.assertTrue(enc.mapping['none']['dave'] == 41)
        self.assertIn('A_nan', enc.mapping['na_categorical'])
        self.assertTrue(enc.mapping['na_categorical']['A_nan'] == 44)

    def test_count_normalize_bool(self):
        """Test the normalize bool on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=26,
            normalize=True
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._normalize)
        self.assertTrue(out['none'].round(5).isin([0.31, 0.41, 0.28]).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([0.28, 0.44], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertTrue(enc.mapping is not None)

    def test_count_normalize_dict(self):
        """Test the normalize dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=26,
            normalize={
                'none': True, 'na_categorical': False
            }
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._normalize)
        self.assertTrue(out['none'].round(5).isin([0.31, 0.41, 0.28]).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isna().sum() == 0)
        self.assertTrue(np.isin([28, 44], out['na_categorical'].unique()).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertTrue(enc.mapping is not None)

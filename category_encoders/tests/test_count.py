import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np
import pandas as pd
import category_encoders as encoders


X = pd.DataFrame({
    'none': [
        'A', 'A', 'B', None, None, 'C', None, 'C', None, 'B',
        'A', 'A', 'C', 'B', 'B', 'A', 'A', None, 'B', None
    ],
    'na_categorical': [
        'A', 'A', 'C', 'A', 'B', 'C', 'C', 'A', np.nan, 'B', 'A',
        'C', 'C', 'A', 'B', 'C', np.nan, 'A', np.nan, np.nan
    ]
})

X_t = pd.DataFrame({
    'none': [
        'A', 'C', None, 'B', 'C', 'C', None, None, 'A',
        'A', 'C', 'A', 'B', 'A', 'A'
    ],
    'na_categorical': [
        'C', 'C', 'A', 'B', 'C', 'A', np.nan, 'B', 'A', 'A',
        'B', np.nan, 'A', np.nan, 'A'
    ]
})

class TestCountEncoder(TestCase):

    def test_count_defaults(self):
        """Test the defaults are working as expected on 'none' and 'categorical' 
        which are the most extreme edge cases for the count encoder."""
        enc = encoders.CountEncoder(verbose=1)
        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(pd.Series([5, 3, 6]).isin(out['none'].unique()).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isnull().sum() == 0)
        self.assertTrue(pd.Series([6, 3]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(enc.mapping is not None)

    def test_count_handle_missing_string(self):
        """Test the handle_missing string on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            handle_missing='return_nan'
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_missing)
        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isnull().sum() == 3)
        self.assertTrue(pd.Series([6, 7, 3]).isin(out['na_categorical']).all())
        self.assertFalse(pd.Series([4]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isnull().sum() == 3)

    def test_count_handle_missing_dict(self):
        """Test the handle_missing dict on 'none' and 'na_categorical'. 
        We want to see differing behavour between 'none' and 'na_cat' cols."""
        enc = encoders.CountEncoder(
            handle_missing={'na_categorical': 'return_nan'}
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_missing)
        self.assertTrue(pd.Series([5, 3, 6]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isnull().sum() == 0)
        self.assertTrue(pd.Series([6, 7, 3]).isin(out['na_categorical']).all())
        self.assertFalse(pd.Series([4]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isnull().sum() == 3)

    def test_count_handle_unknown_string(self):
        """Test the handle_unknown string  on 'none' and 'na_categorical'.
        The 'handle_missing' must be set to 'return_nan' in order to test
        'handle_unkown' correctly."""
        enc = encoders.CountEncoder(
            handle_missing='return_nan',
            handle_unknown='return_nan',
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._handle_unknown)
        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isnull().sum() == 3)
        self.assertTrue(pd.Series([3, 6, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isnull().sum() == 3)

    def test_count_handle_unknown_dict(self):
        """Test the 'handle_unkown' dict with all non-default options."""
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
        self.assertTrue(pd.Series([6, 5, 3, -1]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (4,))
        self.assertTrue(out['none'].isnull().sum() == 0)
        self.assertTrue(pd.Series([3, 6, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (4,))
        self.assertTrue(out['na_categorical'].isnull().sum() == 3)

    def test_count_min_group_size_int(self):
        """Test the min_group_size int  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(min_group_size=7)

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertTrue(out['none'].unique().shape == (3,))
        self.assertTrue(out['none'].isnull().sum() == 0)
        self.assertIn(np.nan, enc.mapping['none'])
        self.assertTrue(pd.Series([13, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertIn('B_C_nan', enc.mapping['na_categorical'])
        self.assertFalse(np.nan in enc.mapping['na_categorical'])

    def test_count_min_group_size_dict(self):
        """Test the min_group_size dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={'none': 6, 'na_categorical': 7}
        )

        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('none', enc._min_group_size)
        self.assertTrue(pd.Series([6, 8]).isin(out['none']).all())
        self.assertEqual(out['none'].unique().shape[0], 2)
        self.assertTrue(out['none'].isnull().sum() == 0)
        self.assertIn(np.nan, enc.mapping['none'])
        self.assertTrue(pd.Series([13, 7]).isin(out['na_categorical']).all())
        self.assertTrue(out['na_categorical'].unique().shape == (2,))
        self.assertIn('B_C_nan', enc.mapping['na_categorical'])
        self.assertFalse(np.nan in enc.mapping['na_categorical'])

    def test_count_combine_min_nan_groups_bool(self):
        """Test the min_nan_groups_bool on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=7,
            combine_min_nan_groups=False
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertTrue(pd.Series([6, 5, 3]).isin(out['none']).all())
        self.assertEqual(out['none'].unique().shape[0], 3)
        self.assertEqual(out['none'].isnull().sum(), 0)
        self.assertTrue(pd.Series([9, 7, 4]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 3)
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_combine_min_nan_groups_dict(self):
        """Test the combine_min_nan_groups dict  on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={
                'none': 6,
                'na_categorical': 7
            },
            combine_min_nan_groups={
                'none': 'force',
                'na_categorical': False
            }
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._combine_min_nan_groups)
        self.assertTrue(pd.Series([14, 6]).isin(out['none']).all())
        self.assertEqual(out['none'].unique().shape[0], 2)
        self.assertEqual(out['none'].isnull().sum(), 0)
        self.assertTrue(pd.Series([9, 7, 4]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 3)
        self.assertTrue(enc.mapping is not None)
        self.assertIn(np.nan, enc.mapping['na_categorical'])

    def test_count_min_group_name_string(self):
        """Test the min_group_name string on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=6,
            min_group_name='dave'
        )

        enc.fit(X)

        self.assertIn('dave', enc.mapping['none'])
        self.assertEqual(enc.mapping['none']['dave'], 8)
        self.assertIn('dave', enc.mapping['na_categorical'])
        self.assertEqual(enc.mapping['na_categorical']['dave'], 7)

    def test_count_min_group_name_dict(self):
        """Test the min_group_name dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size={
                'none': 6, 'na_categorical': 6
            },
            min_group_name={
                'none': 'dave', 'na_categorical': None
            }
        )

        enc.fit(X)

        self.assertIn('none', enc._min_group_name)
        self.assertIn('dave', enc.mapping['none'])
        self.assertEqual(enc.mapping['none']['dave'], 8)
        self.assertIn('B_nan', enc.mapping['na_categorical'])
        self.assertEqual(enc.mapping['na_categorical']['B_nan'], 7)

    def test_count_normalize_bool(self):
        """Test the normalize bool on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=6,
            normalize=True
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._normalize)
        self.assertTrue(out['none'].round(5).isin([0.3, 0.4]).all())
        self.assertEqual(out['none'].unique().shape[0], 2)
        self.assertEqual(out['none'].isnull().sum(), 0)
        self.assertTrue(pd.Series([0.3, 0.35]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 2)
        self.assertTrue(enc.mapping is not None)

    def test_count_normalize_dict(self):
        """Test the normalize dict on 'none' and 'na_categorical'."""
        enc = encoders.CountEncoder(
            min_group_size=7,
            normalize={
                'none': True, 'na_categorical': False
            }
        )

        enc.fit(X)
        out = enc.transform(X_t)

        self.assertIn('none', enc._normalize)
        self.assertTrue(out['none'].round(5).isin([0.3 , 0.15, 0.25]).all())
        self.assertEqual(out['none'].unique().shape[0], 3)
        self.assertEqual(out['none'].isnull().sum(), 0)
        self.assertTrue(pd.Series([13, 7]).isin(out['na_categorical']).all())
        self.assertEqual(out['na_categorical'].unique().shape[0], 2)
        self.assertTrue(enc.mapping is not None)

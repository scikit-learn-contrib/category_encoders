"""Tests for the MultiHotEncoder."""

from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd


class TestMultiHotEncoder(TestCase):
    """Tests for the delimiter-split multi-hot encoder."""

    def test_basic_multi_hot(self):
        """A cell with several items activates one binary column per item."""
        X = pd.DataFrame({'topic': ['math|physics', 'math', 'physics|art', 'art']})
        enc = encoders.MultiHotEncoder(use_cat_names=True).fit(X)
        out = enc.transform(X)
        expected = pd.DataFrame(
            {
                'topic_math': [1, 1, 0, 0],
                'topic_physics': [1, 0, 1, 0],
                'topic_art': [0, 0, 1, 1],
            }
        )
        pd.testing.assert_frame_equal(out, expected, check_dtype=False)
        self.assertEqual(len(enc.get_feature_names_out()), 3)

    def test_delimiter_parameter(self):
        """The delimiter is configurable and may span several characters."""
        X = pd.DataFrame({'c': ['a;b', 'b', 'a']})
        enc = encoders.MultiHotEncoder(delimiter=';', use_cat_names=True).fit(X)
        self.assertEqual(list(enc.get_feature_names_out()), ['c_a', 'c_b'])

        X2 = pd.DataFrame({'c': ['a && b', 'b']})
        enc2 = encoders.MultiHotEncoder(delimiter=' && ', use_cat_names=True).fit(X2)
        self.assertEqual(list(enc2.get_feature_names_out()), ['c_a', 'c_b'])

    def test_empty_items_are_dropped(self):
        """Empty fragments produced by consecutive delimiters are dropped."""
        X = pd.DataFrame({'c': ['a||b', 'a|', '|a']})
        enc = encoders.MultiHotEncoder(use_cat_names=True).fit(X)
        out = enc.transform(X)
        self.assertEqual(list(enc.get_feature_names_out()), ['c_a', 'c_b'])
        self.assertEqual(out.iloc[0].tolist(), [1, 1])
        self.assertEqual(out.iloc[1].tolist(), [1, 0])
        self.assertEqual(out.iloc[2].tolist(), [1, 0])

        X2 = pd.DataFrame({'c': ['a||b', 'a|', '|b']})
        out2 = encoders.MultiHotEncoder(use_cat_names=True).fit_transform(X2)
        self.assertEqual(out2.iloc[2].tolist(), [0, 1])

    def test_whitespace_is_stripped(self):
        """Whitespace around items is stripped before fitting."""
        X = pd.DataFrame({'c': ['a | b', '  a  ', 'b']})
        enc = encoders.MultiHotEncoder(use_cat_names=True).fit(X)
        self.assertEqual(list(enc.get_feature_names_out()), ['c_a', 'c_b'])
        self.assertEqual(enc.transform(X).iloc[0].tolist(), [1, 1])

    def test_empty_and_delimiter_only_cells(self):
        """Cells without any item produce all-zero rows, not errors."""
        X = pd.DataFrame({'c': ['a', '', '|', '   ', 'a']})
        enc = encoders.MultiHotEncoder(use_cat_names=True).fit(X)
        out = enc.transform(X)
        self.assertEqual(list(enc.get_feature_names_out()), ['c_a'])
        self.assertEqual(out.iloc[0].tolist(), [1])
        self.assertEqual(out.iloc[1].tolist(), [0])
        self.assertEqual(out.iloc[2].tolist(), [0])
        self.assertEqual(out.iloc[3].tolist(), [0])

    def test_delimiter_inside_cell_is_always_a_separator(self):
        """The split is unambiguous: every delimiter occurrence separates items.

        MultiHotEncoder cannot know that 'Smith, John' was meant as one item,
        so a cell containing the delimiter always splits into several items
        and a delimiter-only cell yields no item at all.
        """
        X = pd.DataFrame({'name': ['Smith, John', 'Doe, Jane']})
        enc = encoders.MultiHotEncoder(delimiter=',', use_cat_names=True).fit(X)
        out = enc.transform(X)
        self.assertEqual(
            sorted(enc.get_feature_names_out()),
            sorted(['name_Doe', 'name_Jane', 'name_John', 'name_Smith']),
        )
        self.assertEqual(out.iloc[0].tolist(), [1, 1, 0, 0])
        self.assertEqual(out.iloc[1].tolist(), [0, 0, 1, 1])

        X2 = pd.DataFrame({'c': ['|']})
        enc2 = encoders.MultiHotEncoder(use_cat_names=True).fit(X2)
        self.assertEqual(list(enc2.get_feature_names_out()), [])
        self.assertEqual(enc2.transform(X2).shape[1], 0)

    def test_duplicate_items_in_one_cell(self):
        """Repeated items in one cell still produce a binary 1, not a count."""
        X = pd.DataFrame({'c': ['a|a|a', 'a']})
        enc = encoders.MultiHotEncoder(use_cat_names=True).fit(X)
        out = enc.transform(X)
        self.assertEqual(out.iloc[0].tolist(), [1])
        self.assertEqual(out.iloc[1].tolist(), [1])

    def test_unknown_value_all_zeros(self):
        """handle_unknown='value' encodes unknown items as all-zeros."""
        train = pd.DataFrame({'c': ['a', 'b']})
        test = pd.DataFrame({'c': ['a', 'zzz', 'a|zzz']})
        enc = encoders.MultiHotEncoder(handle_unknown='value', use_cat_names=True).fit(train)
        out = enc.transform(test)
        self.assertEqual(list(out.columns), ['c_a', 'c_b'])
        self.assertEqual(out.iloc[0].tolist(), [1, 0])
        self.assertEqual(out.iloc[1].tolist(), [0, 0])
        self.assertEqual(out.iloc[2].tolist(), [1, 0])

    def test_unknown_indicator(self):
        """handle_unknown='indicator' adds a column that lights up for new items."""
        train = pd.DataFrame({'c': ['a', 'b']})
        test = pd.DataFrame({'c': ['a', 'zzz', 'a|zzz']})
        enc = encoders.MultiHotEncoder(handle_unknown='indicator', use_cat_names=True).fit(train)
        out = enc.transform(test)
        self.assertEqual(list(out.columns), ['c_a', 'c_b', 'c_-1'])
        self.assertEqual(out.iloc[0].tolist(), [1, 0, 0])
        self.assertEqual(out.iloc[1].tolist(), [0, 0, 1])
        self.assertEqual(out.iloc[2].tolist(), [1, 0, 1])

    def test_unknown_return_nan(self):
        """handle_unknown='return_nan' encodes a row with an unknown item as NaN."""
        train = pd.DataFrame({'c': ['a', 'b']})
        test = pd.DataFrame({'c': ['a', 'zzz']})
        enc = encoders.MultiHotEncoder(handle_unknown='return_nan', use_cat_names=True).fit(train)
        out = enc.transform(test)
        self.assertTrue(out.iloc[0].notna().all())
        self.assertTrue(out.iloc[1].isna().all())

    def test_unknown_error(self):
        """handle_unknown='error' raises at transform time on unseen items."""
        train = pd.DataFrame({'c': ['a', 'b']})
        test = pd.DataFrame({'c': ['a', 'zzz']})
        enc = encoders.MultiHotEncoder(handle_unknown='error').fit(train)
        with self.assertRaisesRegex(ValueError, 'unknown item'):
            enc.transform(test)

    def test_missing_value_treated_as_item(self):
        """By default a missing cell behaves like another valid item."""
        x_placeholder = pd.Series(['a', 'b', 'b', 'c', 'c'])
        x_nan = pd.Series(['a', 'b', 'b', np.nan, np.nan])
        result_placeholder = encoders.MultiHotEncoder(use_cat_names=True).fit_transform(
            x_placeholder, [0, 1, 1, 1, 1]
        )
        result_nan = encoders.MultiHotEncoder(use_cat_names=True).fit_transform(
            x_nan, [0, 1, 1, 1, 1]
        )
        self.assertEqual(list(result_placeholder.columns), ['0_a', '0_b', '0_c'])
        self.assertEqual(list(result_nan.columns), ['0_a', '0_b', '0_nan'])
        np.testing.assert_equal(result_placeholder.to_numpy(), result_nan.to_numpy())

    def test_missing_indicator(self):
        """handle_missing='indicator' adds a column that lights up for NaN cells."""
        train = pd.DataFrame({'c': ['a', 'b']})
        test = pd.DataFrame({'c': ['a', np.nan]})
        enc = encoders.MultiHotEncoder(handle_missing='indicator', use_cat_names=True).fit(train)
        out = enc.transform(test)
        self.assertEqual(list(out.columns), ['c_a', 'c_b', 'c_-2'])
        self.assertEqual(out.iloc[0].tolist(), [1, 0, 0])
        self.assertEqual(out.iloc[1].tolist(), [0, 0, 1])

    def test_missing_ignore(self):
        """handle_missing='ignore' encodes NaN cells as all zeros without a column."""
        train = pd.DataFrame({'c': ['a', 'b']})
        test = pd.DataFrame({'c': ['a', np.nan]})
        enc = encoders.MultiHotEncoder(handle_missing='ignore', use_cat_names=True).fit(train)
        out = enc.transform(test)
        self.assertEqual(list(out.columns), ['c_a', 'c_b'])
        self.assertEqual(out.iloc[1].tolist(), [0, 0])

    def test_missing_return_nan(self):
        """handle_missing='return_nan' encodes NaN cells as NaN at fit and transform."""
        X = pd.DataFrame({'c': ['a', 'b', np.nan]})
        enc = encoders.MultiHotEncoder(handle_missing='return_nan', use_cat_names=True)
        out = enc.fit_transform(X)
        self.assertTrue(out.iloc[2].isna().all())

        test = pd.DataFrame({'c': ['a', np.nan]})
        out_test = enc.transform(test)
        self.assertTrue(out_test.iloc[1].isna().all())

    def test_missing_error(self):
        """handle_missing='error' raises on NaN cells at fit and transform time."""
        X = pd.DataFrame({'c': ['a', np.nan]})
        enc = encoders.MultiHotEncoder(handle_missing='error')
        with self.assertRaises(ValueError):
            enc.fit(X)

        enc2 = encoders.MultiHotEncoder(handle_missing='error').fit(pd.DataFrame({'c': ['a', 'b']}))
        with self.assertRaises(ValueError):
            enc2.transform(pd.DataFrame({'c': ['a', np.nan]}))

    def test_output_width_is_fixed_by_fit(self):
        """New items at transform time cannot change the output width."""
        X = pd.DataFrame({'c': ['a|b', 'b']})
        X_t = pd.DataFrame({'c': ['a|b', 'new1|new2']})
        enc = encoders.MultiHotEncoder(use_cat_names=True).fit(X)
        width_fit = len(enc.get_feature_names_out())
        out = enc.transform(X_t)
        self.assertEqual(out.shape[1], width_fit)
        self.assertEqual(list(enc.get_feature_names_out()), list(out.columns))

    def test_get_feature_names_out_matches_transform(self):
        """get_feature_names_out equals the transform column names exactly."""
        X = pd.DataFrame({'c': ['a|b', 'b|c'], 'd': ['x', 'y']})
        enc = encoders.MultiHotEncoder().fit(X)
        self.assertListEqual(list(enc.get_feature_names_out()), list(enc.transform(X).columns))

    def test_multi_column_layout_preserved(self):
        """Encoded columns replace their input in place, other columns stay."""
        X = pd.DataFrame({'keep': [1, 2], 'c': ['a|b', 'b'], 'other': ['u', 'v']})
        enc = encoders.MultiHotEncoder(cols=['c'], use_cat_names=True).fit(X)
        out = enc.transform(X)
        self.assertEqual(list(out.columns), ['keep', 'c_a', 'c_b', 'other'])

    def test_use_cat_names_deduplicates(self):
        """Colliding item names and sentinel names get a '#' suffix."""
        X = pd.DataFrame({'c': ['-1', 'a']})
        enc = encoders.MultiHotEncoder(use_cat_names=True, handle_unknown='indicator').fit(X)
        out = enc.transform(pd.DataFrame({'c': ['-1', 'zzz']}))
        self.assertEqual(list(out.columns)[:2], ['c_-1', 'c_a'])
        self.assertTrue(out.columns[2].startswith('c_-1#'))
        # the item '-1' lit the item column, the unseen 'zzz' lit the indicator
        self.assertEqual(out.iloc[0].tolist(), [1, 0, 0])
        self.assertEqual(out.iloc[1].tolist(), [0, 0, 1])

    def test_invalid_delimiter(self):
        """An empty or non-string delimiter is rejected at fit time."""
        with self.assertRaises(ValueError):
            encoders.MultiHotEncoder(delimiter='').fit(pd.DataFrame({'c': ['a|b']}))
        with self.assertRaises(ValueError):
            encoders.MultiHotEncoder(delimiter=None).fit(pd.DataFrame({'c': ['a|b']}))

    def test_no_inverse_transform(self):
        """MultiHotEncoder deliberately has no inverse_transform (scope of #161)."""
        self.assertFalse(hasattr(encoders.MultiHotEncoder, 'inverse_transform'))

    def test_numpy_input(self):
        """Numpy input works and yields a numeric output."""
        arr = np.array([['a|b'], ['b'], ['a|c']])
        enc = encoders.MultiHotEncoder()
        out = enc.fit(arr).transform(arr)
        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(out.iloc[0].tolist(), [1, 1, 0])

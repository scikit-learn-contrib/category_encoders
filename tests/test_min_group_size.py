"""Tests for min_group_size lumping: the shared helper and the base-level hooks.

CountEncoder's own lumping is exercised (unchanged) by tests/test_count.py; these
tests cover the extracted helper and the BaseEncoder-level parameters on
non-count encoders.
"""

from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from category_encoders.utils import build_min_group_map
from sklearn.base import clone

import tests.helpers as th

# 20 rows: A=8, B=4, C=2, D=1, missing=5; every group has both target classes
X_LUMP = pd.DataFrame(
    {
        'cat': [
            'A',
            'A',
            'A',
            'A',
            'A',
            'A',
            'A',
            'A',
            'B',
            'B',
            'B',
            'B',
            'C',
            'C',
            'D',
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        'num': list(range(20)),
    }
)
X_LUMP_T = pd.DataFrame({'cat': ['A', 'B', 'C', 'D', np.nan, 'C', 'E'], 'num': list(range(7))})
Y_LUMP = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

GROUP_SIZES = pd.Series({'A': 8, 'B': 4, 'C': 2, 'D': 1, np.nan: 5})


class TestBuildMinGroupMap(TestCase):
    """Unit tests for the pure lumping helper."""

    def test_int_threshold_lumps_small_groups(self):
        """Groups below the threshold are folded into an alphabetically named leftovers group."""
        sizes, lumping_map = build_min_group_map(GROUP_SIZES, 4, None, True)

        self.assertEqual({'C': 'C_D', 'D': 'C_D'}, lumping_map)
        self.assertEqual({'A': 8, 'B': 4, 'C_D': 3, np.nan: 5}, dict(sizes))

    def test_missing_label_joins_into_leftovers_name(self):
        """A missing-values group below the threshold is folded in and named 'nan'."""
        sizes, lumping_map = build_min_group_map(GROUP_SIZES, 6, None, True)

        self.assertEqual(
            {'B': 'B_C_D_nan', 'C': 'B_C_D_nan', 'D': 'B_C_D_nan', np.nan: 'B_C_D_nan'},
            lumping_map,
        )
        self.assertEqual({'A': 8, 'B_C_D_nan': 12}, dict(sizes))

    def test_force_folds_big_missing_group(self):
        """'force' folds the missing group in even when it is above the threshold."""
        sizes = pd.Series({'A': 8, 'B': 2, np.nan: 9})

        sizes_force, map_force = build_min_group_map(sizes, 3, None, 'force')
        self.assertEqual({'B': 'B_nan', np.nan: 'B_nan'}, map_force)
        self.assertEqual({'A': 8, 'B_nan': 11}, dict(sizes_force))

        # without force, the missing group stays on its own and no lumping happens
        sizes_false, map_false = build_min_group_map(sizes, 3, None, False)
        self.assertEqual({}, map_false)
        self.assertEqual({'A': 8, 'B': 2, np.nan: 9}, dict(sizes_false))

    def test_no_lumping_when_all_groups_are_small(self):
        """Lumping requires at least one surviving group, even with 'force'."""
        sizes, lumping_map = build_min_group_map(pd.Series({'A': 2, 'B': 1}), 3, None, True)

        self.assertEqual({}, lumping_map)
        self.assertEqual({'A': 2, 'B': 1}, dict(sizes))

    def test_no_lumping_with_single_small_group(self):
        """A single small group is not folded anywhere."""
        sizes, lumping_map = build_min_group_map(pd.Series({'A': 8, 'B': 4, 'C': 1}), 2, None, True)

        self.assertEqual({}, lumping_map)
        self.assertEqual({'A': 8, 'B': 4, 'C': 1}, dict(sizes))

    def test_custom_leftovers_name(self):
        """A custom min_group_name overrides the joined default."""
        _, lumping_map = build_min_group_map(GROUP_SIZES, 4, 'dave', True)

        self.assertEqual({'C': 'dave', 'D': 'dave'}, lumping_map)

    def test_input_series_is_not_modified(self):
        """The helper is pure: the passed group sizes are left untouched."""
        sizes = pd.Series({'A': 8, 'B': 4, 'C': 2, 'D': 1})
        expected = dict(sizes)

        build_min_group_map(sizes, 4, None, True)

        self.assertEqual(expected, dict(sizes))


class TestMinGroupSizeBase(TestCase):
    """BaseEncoder-level min_group_size / min_group_name / combine_min_nan_groups."""

    def test_target_encoder_merges_rare_groups(self):
        """Lumped encodings equal encoding a manually relabeled column with a plain encoder."""
        enc = encoders.TargetEncoder(min_group_size=4)
        out = enc.fit_transform(X_LUMP[['cat']], Y_LUMP)

        self.assertEqual({'cat': {'C': 'C_D', 'D': 'C_D'}}, enc.min_group_lumping_)

        X_manual = X_LUMP[['cat']].replace({'C': 'C_D', 'D': 'C_D'})
        plain = encoders.TargetEncoder().fit(X_manual, Y_LUMP)
        np.testing.assert_allclose(plain.transform(X_manual)['cat'], out['cat'], rtol=1e-12)

    def test_transform_falls_through_for_unseen_labels(self):
        """Labels unseen at fit are not remapped at transform time."""
        enc = encoders.TargetEncoder(min_group_size=4)
        enc.fit(X_LUMP[['cat']], Y_LUMP)
        out = enc.transform(X_LUMP_T[['cat']])

        self.assertEqual(7, len(out))
        self.assertTrue(out['cat'].notna().all())

    def test_missing_group_semantics(self):
        """True folds a small missing group, 'force' a big one, False never."""
        # missing count is 5; threshold 4 leaves it above the threshold
        enc_true = encoders.WOEEncoder(min_group_size=4)
        enc_true.fit(X_LUMP[['cat']], Y_LUMP)
        self.assertEqual({'C', 'D'}, set(enc_true.min_group_lumping_['cat']))

        enc_force = encoders.WOEEncoder(min_group_size=4, combine_min_nan_groups='force')
        out_force = enc_force.fit(X_LUMP[['cat']], Y_LUMP).transform(X_LUMP_T[['cat']])
        force_keys = set(enc_force.min_group_lumping_['cat'])
        self.assertEqual({'C', 'D'}, {key for key in force_keys if key == key})
        self.assertEqual(3, len(force_keys))  # plus one NaN key
        self.assertEqual(
            out_force.loc[[0, 2, 3, 4], 'cat'].tolist(),  # A, C, D and missing rows
            [
                out_force.loc[0, 'cat'],
                out_force.loc[2, 'cat'],
                out_force.loc[2, 'cat'],
                out_force.loc[2, 'cat'],
            ],
        )

        enc_false = encoders.WOEEncoder(min_group_size=4, combine_min_nan_groups=False)
        out_false = enc_false.fit(X_LUMP[['cat']], Y_LUMP).transform(X_LUMP_T[['cat']])
        self.assertEqual({'C', 'D'}, set(enc_false.min_group_lumping_['cat']))
        self.assertNotEqual(out_false.loc[4, 'cat'], out_false.loc[2, 'cat'])

    def test_ordinal_encoder_shares_code_for_lumped_labels(self):
        """Lumped labels are encoded as one ordinal code."""
        enc = encoders.OrdinalEncoder(min_group_size=4)
        out = enc.fit(X_LUMP[['cat']]).transform(X_LUMP_T[['cat']])
        codes = dict(zip(X_LUMP_T['cat'], out['cat'], strict=True))

        self.assertEqual(4, enc.transform(X_LUMP[['cat']])['cat'].nunique())
        self.assertEqual(codes['C'], codes['D'])
        # lumping is lossy on inverse_transform: the leftovers name comes back
        self.assertEqual('C_D', enc.inverse_transform(out).loc[2, 'cat'])

    def test_one_hot_encoder_creates_single_leftovers_column(self):
        """The leftovers group occupies one output column."""
        enc = encoders.OneHotEncoder(min_group_size=4)
        enc.fit(X_LUMP[['cat']])

        self.assertEqual(4, len(enc.get_feature_names_out()))

    def test_lumping_applies_across_encoders(self):
        """The base-level hook drives the same lumping on every label-mapped encoder."""
        encoders_under_test = [
            encoders.TargetEncoder,
            encoders.MEstimateEncoder,
            encoders.JamesSteinEncoder,
            encoders.WOEEncoder,
            encoders.CatBoostEncoder,
            encoders.LeaveOneOutEncoder,
        ]
        for encoder_cls in encoders_under_test:
            with self.subTest(encoder_name=encoder_cls.__name__):
                enc = encoder_cls(min_group_size=4)
                enc.fit(X_LUMP[['cat']], Y_LUMP)
                out = enc.transform(X_LUMP_T[['cat']])

                self.assertEqual({'C': 'C_D', 'D': 'C_D'}, enc.min_group_lumping_['cat'])
                self.assertEqual(out.loc[2, 'cat'], out.loc[3, 'cat'])

    def test_off_by_default_matches_plain_output(self):
        """Default and explicitly-None parameters produce identical output."""
        X = th.create_dataset(n_rows=100)
        y = pd.Series(np.random.RandomState(42).randn(100) > 0)

        plain = encoders.TargetEncoder().fit(X[['categorical']], y)
        explicit = encoders.TargetEncoder(
            min_group_size=None, min_group_name=None, combine_min_nan_groups=None
        ).fit(X[['categorical']], y)

        np.testing.assert_allclose(
            plain.transform(X[['categorical']])['categorical'],
            explicit.transform(X[['categorical']])['categorical'],
        )

    def test_refit_resets_lumping(self):
        """A refit without min_group_size drops the previously learned lumping."""
        enc = encoders.TargetEncoder(min_group_size=4)
        enc.fit(X_LUMP[['cat']], Y_LUMP)
        self.assertTrue(enc.min_group_lumping_)

        enc.set_params(min_group_size=None)
        enc.fit(X_LUMP[['cat']], Y_LUMP)
        self.assertEqual({}, enc.min_group_lumping_)

    def test_fit_does_not_mutate_the_callers_frame(self):
        """Lumping the training data leaves the passed DataFrame untouched."""
        X = X_LUMP.copy(deep=True)

        encoders.TargetEncoder(min_group_size=4).fit(X, Y_LUMP)

        pd.testing.assert_frame_equal(X_LUMP, X)

    def test_float_threshold_is_a_fraction_of_rows(self):
        """A float min_group_size is resolved as a fraction of the row count."""
        enc = encoders.TargetEncoder(min_group_size=0.25)  # 0.25 * 20 = 5
        enc.fit(X_LUMP[['cat']], Y_LUMP)

        self.assertEqual({'B', 'C', 'D'}, set(enc.min_group_lumping_['cat']))

    def test_parameters_survive_clone(self):
        """The new parameters round-trip through get_params / clone."""
        enc = encoders.TargetEncoder(min_group_size=4, min_group_name='dave')
        params = clone(enc).get_params()

        self.assertEqual(4, params['min_group_size'])
        self.assertEqual('dave', params['min_group_name'])

    def test_invalid_parameters_raise(self):
        """Parameter conflicts raise a ValueError at fit time."""
        cases = [
            (
                {'min_group_name': 'dave'},
                '`min_group_name` only works when `min_group_size` is set.',
            ),
            (
                {'min_group_size': 4, 'combine_min_nan_groups': 'banana'},
                "'combine_min_nan_groups' should be one of: ['force', True, False, None].",
            ),
            (
                {
                    'min_group_size': 4,
                    'combine_min_nan_groups': 'force',
                    'handle_missing': 'return_nan',
                },
                "Cannot have `handle_missing` == 'return_nan' and "
                "'combine_min_nan_groups' == 'force'.",
            ),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as ctx:
                    encoders.TargetEncoder(**kwargs).fit(X_LUMP[['cat']], Y_LUMP)
                self.assertEqual(message, str(ctx.exception))


class TestCountEncoderOptOut(TestCase):
    """CountEncoder keeps its own lumping and opts out of the base-level hooks."""

    def test_count_encoder_does_not_double_lump(self):
        """CountEncoder's fit never triggers the base-level hook."""
        enc = encoders.CountEncoder(min_group_size=7)
        enc.fit(X_LUMP[['cat']])

        self.assertFalse(enc._min_group_hooks_enabled)
        self.assertEqual({}, enc.min_group_lumping_)
        self.assertTrue(enc._min_group_categories)

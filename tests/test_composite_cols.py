"""Tests for composite_cols: joint encoding of column groups on supervised encoders."""

from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd


def make_dataset(with_missing: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Small deterministic dataset with low-cardinality categorical columns."""
    rng = np.random.RandomState(7)
    n_rows = 240
    X = pd.DataFrame(
        {
            'product': rng.choice(['phone', 'laptop', 'tablet'], n_rows),
            'color': rng.choice(['red', 'green', 'blue', 'black'], n_rows),
            'region': rng.choice(['north', 'south'], n_rows),
        }
    )
    if with_missing:
        X.loc[0, 'color'] = np.nan
        X.loc[1, 'product'] = np.nan
    y = pd.Series(rng.binomial(1, 0.5, n_rows), name='target')
    return X, y


def concat_manually(X: pd.DataFrame, members: tuple[str, ...]) -> pd.DataFrame:
    """Build the manually-concatenated frame the composite output must match."""
    X = X.copy()
    name = '|'.join(members)
    X[name] = X[members[0]].astype(str)
    for member in members[1:]:
        X[name] = X[name] + '|' + X[member].astype(str)
    X.loc[X[list(members)].isna().any(axis=1), name] = np.nan
    return X


class TestCompositeCols(TestCase):
    """Unit tests for the composite_cols parameter."""

    def test_joint_equals_manual_concat(self):
        """The composite column must equal a manually joined column, encoded alike."""
        for encoder_cls in (
            encoders.TargetEncoder,
            encoders.WOEEncoder,
            encoders.MEstimateEncoder,
        ):
            with self.subTest(encoder=encoder_cls.__name__):
                X, y = make_dataset()
                composite = encoder_cls(composite_cols=[('product', 'color')])
                result = composite.fit(X, y).transform(X, y)

                X_manual = concat_manually(X, ('product', 'color'))
                manual = encoder_cls(cols=['product|color'])
                expected = manual.fit(X_manual[['product|color']], y).transform(
                    X_manual[['product|color']], y
                )

                np.testing.assert_allclose(
                    result['product|color'].to_numpy(), expected['product|color'].to_numpy()
                )
                self.assertEqual(list(result.columns), ['region', 'product|color'])

    def test_joint_equals_manual_concat_with_missing(self):
        """Missing members become a missing composite value on both sides alike."""
        X, y = make_dataset(with_missing=True)
        composite = encoders.TargetEncoder(composite_cols=[('product', 'color')])
        result = composite.fit(X, y).transform(X, y)

        X_manual = concat_manually(X, ('product', 'color'))
        manual = encoders.TargetEncoder(cols=['product|color'])
        expected = manual.fit(X_manual[['product|color']], y).transform(
            X_manual[['product|color']], y
        )

        np.testing.assert_allclose(
            result['product|color'].to_numpy(), expected['product|color'].to_numpy()
        )

    def test_keep_components_retains_encoded_members(self):
        """keep_components keeps the members encoded individually next to the joint."""
        X, y = make_dataset()
        composite = encoders.TargetEncoder(
            composite_cols=[('product', 'color')], keep_components=True
        )
        result = composite.fit(X, y).transform(X, y)
        self.assertEqual(sorted(result.columns), ['color', 'product', 'product|color', 'region'])

        individual = encoders.TargetEncoder(cols=['product'])
        expected = individual.fit(X[['product']], y).transform(X[['product']], y)
        np.testing.assert_allclose(result['product'].to_numpy(), expected['product'].to_numpy())

    def test_feature_names(self):
        """Composite names come from feature_names_out_ automatically."""
        X, y = make_dataset()
        composite = encoders.TargetEncoder(composite_cols=[('product', 'color')])
        composite.fit(X, y)
        np.testing.assert_array_equal(
            composite.get_feature_names_out(), ['region', 'product|color']
        )

        kept = encoders.TargetEncoder(composite_cols=[('product', 'color')], keep_components=True)
        kept.fit(X, y)
        np.testing.assert_array_equal(
            kept.get_feature_names_out(), ['product', 'color', 'region', 'product|color']
        )

    def test_unknown_combination_gets_prior(self):
        """An unseen member combination falls back to the prior like any unknown value."""
        X, y = make_dataset()
        composite = encoders.TargetEncoder(composite_cols=[('product', 'color')])
        composite.fit(X, y)

        X_new = X.head(10).copy()
        X_new['product'] = 'telegraph'  # unseen member value -> unseen combination
        result = composite.transform(X_new)

        np.testing.assert_allclose(result['product|color'].to_numpy(), composite._mean, rtol=1e-12)

    def test_separator_collision_raises(self):
        """A '|' in the data that collapses two combinations must be rejected."""
        X, y = make_dataset()
        X.loc[0] = ['a|b', 'c', 'north']
        X.loc[1] = ['a', 'b|c', 'south']  # both rows join to 'a|b|c'
        composite = encoders.TargetEncoder(composite_cols=[('product', 'color')])
        with self.assertRaises(ValueError):
            composite.fit(X, y)

    def test_refit_resets_composite_state(self):
        """Refitting rebuilds the composite state instead of appending to it."""
        X, y = make_dataset()
        composite = encoders.TargetEncoder(composite_cols=[('product', 'color')])
        composite.fit(X, y)

        X2, y2 = make_dataset()
        composite.fit(X2, y2)

        self.assertEqual(list(composite.composite_members_), ['product|color'])
        self.assertEqual(list(composite.get_feature_names_out()), ['region', 'product|color'])

    def test_composite_with_explicit_cols(self):
        """Composites work independently of the cols selection."""
        X, y = make_dataset()
        composite = encoders.TargetEncoder(cols=['region'], composite_cols=[('product', 'color')])
        result = composite.fit(X, y).transform(X, y)
        self.assertEqual(list(result.columns), ['region', 'product|color'])

    def test_default_composite_cols_is_parity(self):
        """Without composites the encoder behaves exactly as before."""
        X, y = make_dataset()
        base = encoders.TargetEncoder().fit(X, y).transform(X, y)
        off = encoders.TargetEncoder(composite_cols=None).fit(X, y).transform(X, y)
        pd.testing.assert_frame_equal(base, off)

    def test_unsupervised_encoder_rejects_composites(self):
        """composite_cols is a supervised-encoder feature; unsupervised use is rejected."""
        X, y = make_dataset()
        ordinal = encoders.OrdinalEncoder()
        ordinal.composite_cols = [('product', 'color')]  # not exposed on unsupervised inits
        with self.assertRaises(ValueError):
            ordinal.fit(X, y)

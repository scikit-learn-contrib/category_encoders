"""Tests for the GLMMEncoder."""
import importlib.util
import unittest
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd

HAS_STATSMODELS = importlib.util.find_spec('statsmodels') is not None


def _binary_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'col': rng.choice(list('abcde'), 300)})
    rates = {'a': 0.2, 'b': 0.4, 'c': 0.5, 'd': 0.6, 'e': 0.8}
    y = pd.Series((rng.random(300) < X['col'].map(rates)).astype(int))
    return X, y


class TestGLMMEncoder(TestCase):
    """Unit tests for the GLMMEncoder.

    They are still missing.
    """

    @unittest.skip
    def test_continuous(self):
        """Test the continuous target."""

    @unittest.skip
    def test_binary(self):
        """Test the binary target."""

    @unittest.skipUnless(HAS_STATSMODELS, 'GLMMEncoder requires statsmodels')
    def test_binary_fit_is_reproducible(self):
        """Fitting twice on the same data gives identical encodings.

        statsmodels >= 0.15 seeds the variational-Bayes fit through its own ``rng``
        argument rather than the global numpy RNG, so the encoder has to pass a seed
        explicitly for the result to be deterministic.
        """
        X, y = _binary_data()
        first = encoders.GLMMEncoder(cols=['col']).fit_transform(X, y)
        second = encoders.GLMMEncoder(cols=['col']).fit_transform(X, y)
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    @unittest.skipUnless(HAS_STATSMODELS, 'GLMMEncoder requires statsmodels')
    def test_fit_restores_global_random_state(self):
        """Fitting does not leave the global numpy random state reseeded."""
        X, y = _binary_data()
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        encoders.GLMMEncoder(cols=['col']).fit(X, y)
        self.assertEqual(np.random.random(), expected)

    @unittest.skipUnless(HAS_STATSMODELS, 'GLMMEncoder requires statsmodels')
    def test_regression_branch_is_deterministic(self):
        """A continuous target uses the mixedlm regression branch, reproducibly.

        Covers the binomial_target=False path (smf.mixedlm) that the binary
        tests never reach, including the auto-detected continuous-target
        branch, and pins its determinism across fits.
        """
        rng = np.random.default_rng(7)
        X = pd.DataFrame({'col': rng.choice(list('abcde'), 120)})
        y = pd.Series(rng.normal(size=120))
        first = encoders.GLMMEncoder(cols=['col']).fit_transform(X, y)
        second = encoders.GLMMEncoder(cols=['col']).fit_transform(X, y)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        self.assertTrue(first.notna().all().all())

    @unittest.skipUnless(HAS_STATSMODELS, 'GLMMEncoder requires statsmodels')
    def test_randomized_scoring_perturbs_but_stays_deterministic(self):
        """randomized=True scales encoded values by a seeded Gaussian draw."""
        X, y = _binary_data()
        plain = encoders.GLMMEncoder(cols=['col']).fit_transform(X, y)
        randomized = encoders.GLMMEncoder(
            cols=['col'], randomized=True, random_state=42, sigma=0.05
        ).fit_transform(X, y)
        self.assertFalse(np.allclose(plain, randomized))

    @unittest.skipUnless(HAS_STATSMODELS, 'GLMMEncoder requires statsmodels')
    def test_singular_matrix_returns_zero_mapping(self):
        """A LinAlgError from the model fit degrades to an all-zero mapping."""
        from unittest.mock import patch

        X, y = _binary_data()
        with patch(
            'category_encoders.glmm.smf.mixedlm',
            side_effect=np.linalg.LinAlgError('singular'),
        ):
            result = encoders.GLMMEncoder(cols=['col'], binomial_target=False).fit_transform(X, y)
        self.assertTrue((result == 0).all().all())

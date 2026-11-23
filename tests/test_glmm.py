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

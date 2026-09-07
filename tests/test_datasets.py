"""Tests for the bundled dataset loaders."""
from unittest import TestCase

from category_encoders.datasets._base import load_compass, load_postcodes


class TestDatasetLoaders(TestCase):
    """Contract tests for the packaged CSV loaders."""

    def test_load_compass(self):
        """load_compass returns the documented 16-row compass frame.

        Mutants in category_encoders/datasets/_base.py that drop the packaged
        file path or rename columns break these contracts.
        """
        X, y = load_compass()
        self.assertEqual(X.shape, (16, 3))
        self.assertEqual(list(X.columns), ['index', 'compass', 'HIER_compass_1'])
        self.assertEqual(y.name, 'target')
        self.assertEqual(len(y), 16)

    def test_load_postcodes_default_target_is_binary(self):
        """load_postcodes defaults to the binary target; X excludes targets.

        Mutants of the target_type parameter (mutmut_1/2) and of the
        startswith filter (mutmut_22/23) leak target columns into X or
        rename y.
        """
        X, y = load_postcodes()
        self.assertEqual(y.name, 'target_binary')
        self.assertFalse(any(col.startswith('target') for col in X.columns))
        self.assertEqual(len(X), 100)

    def test_load_postcodes_non_binary(self):
        """The target_type argument selects the requested target column."""
        X, y = load_postcodes(target_type='non_binary')
        self.assertEqual(y.name, 'target_non_binary')
        self.assertFalse(any(col.startswith('target') for col in X.columns))

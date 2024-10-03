"""Tests for cat boost encoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd


class TestCatBoostEncoder(TestCase):
    """Tests for the CatBoostEncoder."""

    def test_cat_boost(self):
        """Test the CatBoostEncoder."""
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A']})
        y = pd.Series([1, 0, 1, 0, 1])
        enc = encoders.CatBoostEncoder()
        obtained = enc.fit_transform(X, y)
        self.assertEqual(
            list(obtained['col1']),
            [0.6, 0.6, 0.6 / 2, 0.6, 1.6 / 2],
            'The nominator is incremented by the prior. The denominator by 1.',
        )

        # For testing set, use statistics calculated on all the training data.
        # See: CatBoost: unbiased boosting with categorical features, page 4.
        X_t = pd.DataFrame({'col1': ['B', 'B', 'A']})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [1.6 / 3, 1.6 / 3, 2.6 / 3])

    def test_cat_boost_missing(self):
        """Test the CatBoostEncoder with missing values.

        Should impute according to cat boost for missing values.
        """
        X = pd.DataFrame({'col1': ['A', 'B', 'B', 'C', 'A', np.nan, np.nan, np.nan]})
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        enc = encoders.CatBoostEncoder(handle_missing='value')
        obtained = enc.fit_transform(X, y)
        self.assertEqual(
            list(obtained['col1']),
            [0.5, 0.5, 0.5 / 2, 0.5, 1.5 / 2, 0.5, 0.5 / 2, 1.5 / 3],
            'We treat None as another category.',
        )

        X_t = pd.DataFrame({'col1': ['B', 'B', 'A', np.nan]})
        obtained = enc.transform(X_t)
        self.assertEqual(list(obtained['col1']), [1.5 / 3, 1.5 / 3, 2.5 / 3, 1.5 / 4])

    def test_cat_boost_reference(self):
        """Test a specific case mentioned on catboost website.

        The reference is from:
          https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html
        paragraph:
            Transforming categorical features to numerical features in classification
        as obtained on 17 Aug 2019.
        """
        X = pd.DataFrame({'col1': ['rock', 'indie', 'rock', 'rock', 'pop', 'indie', 'rock']})
        y = pd.Series([0, 0, 1, 1, 1, 0, 0])
        enc = encoders.CatBoostEncoder()
        obtained = enc.fit_transform(X, y)
        # Since we do not support prior passing, we replace the prior in the reference = 0.05
        # with the sample prior = 3/7.
        prior = 3.0 / 7
        self.assertEqual(
            list(obtained['col1']),
            [prior, prior, prior / 2, (1 + prior) / 3, prior, prior / 2, (2 + prior) / 4],
        )

    def test_cat_boost_reference_2(self):
        """Test another reference case stated in a video.

        The reference is from:
          https://www.youtube.com/watch?v=hqYQ8Yj9vB0
        time:
            35:03
        as obtained on 21 Aug 2019.
        Note: they have an error at line [smooth 6 4.3 4.1]. It should be [smooth 6 4 4.1 3.9]
        """
        X = pd.DataFrame(
            {'col1': ['fuzzy', 'soft', 'smooth', 'fuzzy', 'smooth', 'soft', 'smooth', 'smooth']}
        )
        y = pd.Series([4, 1, 4, 3, 6, 0, 7, 5])
        enc = encoders.CatBoostEncoder()
        obtained = enc.fit_transform(X, y)
        prior = 30.0 / 8
        self.assertEqual(
            list(obtained['col1']),
            [
                prior,
                prior,
                prior,
                (4 + prior) / 2,
                (4 + prior) / 2,
                (1 + prior) / 2,
                (10 + prior) / 3,
                (17 + prior) / 4,
            ],
        )

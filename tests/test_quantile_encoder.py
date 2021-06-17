import unittest

import pandas as pd
import category_encoders as encoders
import numpy as np


class TestQuantileEncoder(unittest.TestCase):
    """Tests for percentile encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable"""

        self.df = pd.DataFrame(
            {"categories": ["a", "b", "c", "a", "b", "c", "a", "b"]}
        )
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])

    def test_median_works(self):
        """
        Expected output of percentile 50 in df:
            - a median is 4 (a values are 1, 4, 6)
            - b median is 5 (b values are 2, 5, 7)
            - c median is 0 (c values are 0)
        """

        expected_output_median = pd.DataFrame(
            {"categories": [4.0, 5, 0, 4, 5, 0, 4, 5]}
        )

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=0.5, m=0.0).fit_transform(
                self.df, self.target
            ),
            expected_output_median,
        )

    def test_max_works(self):
        """
        Expected output of percentile 100 in df:
            - a max is 6
            - b max is 7
            - c max is 0
        """
        expected_output_max = pd.DataFrame(
            {"categories": [6.0, 7, 0, 6, 7, 0, 6, 7]}
        )

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=1, m=0.0).fit_transform(
                self.df, self.target
            ),
            expected_output_max,
        )

    def test_new_category(self):
        """
        The global median of the target is 3. If new categories are passed to
        the transformer, then the output should be 3
        """
        transformer_median = encoders.QuantileEncoder(quantile=0.5, m=0.0)
        transformer_median.fit(self.df, self.target)

        new_df = pd.DataFrame({"categories": ["d", "e"]})

        new_medians = pd.DataFrame({"categories": [3.0, 3.0]})

        pd.testing.assert_frame_equal(
            transformer_median.transform(new_df), new_medians
        )

"""Tests for quantile encoder."""
import unittest

import category_encoders as encoders
import numpy as np
import pandas as pd


class TestQuantileEncoder(unittest.TestCase):
    """Tests for percentile encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable."""
        self.df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']})
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])

    def test_median_works(self):
        """Test that median encoder works.

        Expected output of percentile 50 in df:
            - a median is 4 (a values are 1, 4, 6)
            - b median is 5 (b values are 2, 5, 7)
            - c median is 0 (c values are 0)
        """
        expected_output_median = pd.DataFrame({'categories': [4.0, 5, 0, 4, 5, 0, 4, 5]})

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=0.5, m=0.0).fit_transform(self.df, self.target),
            expected_output_median,
        )

    def test_max_works(self):
        """Test that maximum (=percentile 100) encoder works.

        Expected output of percentile 100 in df:
            - a max is 6
            - b max is 7
            - c max is 0
        """
        expected_output_max = pd.DataFrame({'categories': [6.0, 7, 0, 6, 7, 0, 6, 7]})

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=1, m=0.0).fit_transform(self.df, self.target),
            expected_output_max,
        )

    def test_new_category(self):
        """Test that unknown values are encoded with global mean.

        The global median of the target is 3. If new categories are passed to
        the transformer, then the output should be 3
        """
        transformer_median = encoders.QuantileEncoder(quantile=0.5, m=0.0)
        transformer_median.fit(self.df, self.target)

        new_df = pd.DataFrame({'categories': ['d', 'e']})

        new_medians = pd.DataFrame({'categories': [3.0, 3.0]})

        pd.testing.assert_frame_equal(transformer_median.transform(new_df), new_medians)

    def test_unique_column_collapses_to_prior(self):
        """Test that a fully unique (id-like) column encodes every category to the prior.

        With the default m=1.0, singleton levels used to blend their own target
        quantile 50/50 with the prior, which made id-like columns predictive of
        the label. See issue #327.
        """
        df = pd.DataFrame({'id': [f'row{i}' for i in range(6)]})
        target = np.array([1, 0, 1, 0, 1, 0])

        for quantile in [0.25, 0.5, 0.75]:
            enc = encoders.QuantileEncoder(quantile=quantile)
            result = enc.fit_transform(df, target)
            np.testing.assert_allclose(
                result['id'],
                np.quantile(target, quantile),
                err_msg=f'quantile={quantile}: a unique column must encode to the prior',
            )
            self.assertTrue(all(result.var() < 0.001))

    def test_column_with_repeated_categories_is_not_collapsed(self):
        """Test that the collapse changes outputs only for fully unique columns.

        The fixture mixes repeated levels with a singleton ('d', target 7).
        The column as a whole is not unique, so no level collapses to the
        prior 3.0: 'a', 'b' and 'c' keep their smoothed statistics and the
        singleton 'd' keeps its own smoothed value 5.0. See issue #327.
        """
        df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'd']})
        target = np.array([1, 2, 0, 4, 5, 0, 6, 7])
        expected_output = pd.DataFrame(
            {'categories': [3.75, 10 / 3, 1.0, 3.75, 10 / 3, 1.0, 3.75, 5.0]}
        )

        pd.testing.assert_frame_equal(
            encoders.QuantileEncoder(quantile=0.5).fit_transform(df, target),
            expected_output,
        )

    def test_m_still_controls_shrinkage_after_collapse(self):
        """Test that m keeps controlling shrinkage for non-unique columns.

        A large m pulls every level of a repeated-category column close to
        the prior (3.0 for this fixture).
        """
        result = encoders.QuantileEncoder(quantile=0.5, m=1000).fit_transform(self.df, self.target)

        np.testing.assert_allclose(result['categories'], 3.0, atol=0.01)


class TestSummaryEncoder(unittest.TestCase):
    """Tests for summary encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable."""
        self.df = pd.DataFrame({'categories': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b']})
        self.target = np.array([1, 2, 0, 4, 5, 0, 6, 7])
        self.col = 'categories'

    def assert_same_quantile(self, quantile):
        """Check the summary encoder with a single quantile coincides with the quantile encoder."""
        quantile_results = encoders.QuantileEncoder(
            cols=[self.col], quantile=quantile
        ).fit_transform(self.df, self.target)

        summary_results = encoders.SummaryEncoder(
            cols=[self.col], quantiles=[quantile]
        ).fit_transform(self.df, self.target)

        percentile = round(quantile * 100)
        col_name = str(self.col) + '_' + str(percentile)
        np.testing.assert_allclose(
            quantile_results[self.col].values,
            summary_results[col_name].values,
        )

    def test_several_quantiles(self):
        """Check that all quantiles of the QE are in the summary encoder."""
        for quantile in [0.1, 0.5, 0.9]:
            self.assert_same_quantile(quantile)

    def test_several_quantiles_reverse(self):
        """Check that all quantiles of summary encoder are in the quantile encoder."""
        quantile_list = [0.2, 0.1, 0.8]

        summary_results = encoders.SummaryEncoder(
            cols=[self.col], quantiles=quantile_list
        ).fit_transform(self.df, self.target)

        for quantile in quantile_list:
            quantile_results = encoders.QuantileEncoder(
                cols=[self.col], quantile=quantile
            ).fit_transform(self.df, self.target)

            percentile = round(quantile * 100)
            col_name = str(self.col) + '_' + str(percentile)

            np.testing.assert_allclose(
                quantile_results[self.col].values,
                summary_results[col_name].values,
            )

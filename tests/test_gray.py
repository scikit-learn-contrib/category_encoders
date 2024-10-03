"""Unit tests for the GrayEncoder."""
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd


class TestGrayEncoder(TestCase):
    """Unit tests for the GrayEncoder."""

    def test_gray_sorting(self):
        """Test the GrayEncoder sorting."""
        data = np.array(['ba', 'ba', 'aa'])
        out = encoders.GrayEncoder().fit_transform(data)
        expected = pd.DataFrame([[1, 1], [1, 1], [0, 1]], columns=['0_0', '0_1'])
        pd.testing.assert_frame_equal(out, expected)

    def test_gray_mapping(self):
        """Test the GrayEncoder mapping."""
        train_data = pd.DataFrame()
        train_data['cat_col'] = np.array([4, 9, 6, 7, 7, 9])
        train_data['other_col'] = range(train_data.shape[0])
        encoder = encoders.GrayEncoder(cols=['cat_col'])
        encoder.fit(train_data)

        expected_ordinal_mapping = {4.0: 1, 9.0: 2, 6.0: 3, 7.0: 4, 'nan': -2}
        expected_mapping = pd.DataFrame(
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            columns=[f'cat_col_{i}' for i in range(3)],
            index=[1, 3, 4, 2, -1, -2],
        )
        self.assertEqual(len(encoder.mapping), 1)
        self.assertEqual(len(encoder.mapping[0].keys()), 2)

        actual_ordinal_encoding = encoder.ordinal_encoder.mapping[0]['mapping']
        actual_ordinal_encoding.index = actual_ordinal_encoding.index.fillna('nan')
        self.assertDictEqual(actual_ordinal_encoding.to_dict(), expected_ordinal_mapping)
        pd.testing.assert_frame_equal(encoder.mapping[0]['mapping'], expected_mapping)

        train_transformed = encoder.transform(train_data)
        train_data['cat_col'] = np.array([4, 9, 6, 7, 7, 9])
        expected_train_transformed = [
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 2],
            [0, 1, 0, 3],
            [0, 1, 0, 4],
            [1, 1, 0, 5],
        ]
        expected_train_transformed = pd.DataFrame(
            expected_train_transformed,
            columns=[f'cat_col_{i}' for i in range(3)] + ['other_col'],
            index=train_data.index,
        )
        pd.testing.assert_frame_equal(train_transformed, expected_train_transformed)
        test_data = pd.DataFrame()
        test_data['cat_col'] = np.array([4, 3, None, np.nan])
        test_data['other_col'] = range(test_data.shape[0])
        expected_test_transformed = [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
        ]
        expected_test_transformed = pd.DataFrame(
            expected_test_transformed,
            columns=[f'cat_col_{i}' for i in range(3)] + ['other_col'],
            index=test_data.index,
        )
        test_transformed = encoder.transform(test_data)
        pd.testing.assert_frame_equal(test_transformed, expected_test_transformed)

    def test_gray_code(self):
        """Test the Gray code generation."""
        input_expected_output = {
            (0, 0): [0],
            (0, 1): [0],
            (0, 3): [0, 0, 0],
            (1, 1): [1],
            (1, 3): [0, 0, 1],
            (2, 2): [1, 1],
            (13, 4): [1, 0, 1, 1],
            (13, 6): [0, 0, 1, 0, 1, 1],
        }
        for test_input, expected_output in input_expected_output.items():
            n, n_bits = test_input
            out = encoders.GrayEncoder.gray_code(n, n_bits)
            self.assertEqual(out, expected_output)

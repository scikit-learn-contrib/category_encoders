"""Unit tests for the SumEncoder."""
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd

a_encoding = [1, 0]
b_encoding = [0, 1]
c_encoding = [-1, -1]


class TestSumEncoder(TestCase):
    """Unit tests for the SumEncoder."""

    def test_unknown_and_missing(self):
        """Test the SumEncoder with the handle unknown = 'value' strategy."""
        train = ['A', 'B', 'C']

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        dim_1_test = ['A', 'D', 'E']
        dim_1_expected = [a_encoding, [0, 0], [0, 0]]
        dim_2_test = ['B', 'D', 'E']
        dim_2_expected = [b_encoding, [0, 0], [0, 0]]
        dim_3_test = ['A', 'B', 'C', None]
        dim_3_expected = [a_encoding, b_encoding, c_encoding, [0, 0]]

        dim_4_test = ['D', 'B', 'C', None]
        dim_4_expected = [[0, 0], b_encoding, c_encoding, [0, 0]]
        cases = {"should preserve dimension 1": (dim_1_test, dim_1_expected),
                 "should preserve dimension 2": (dim_2_test, dim_2_expected),
                 "should preserve dimension 3": (dim_3_test, dim_3_expected),
                 "should preserve dimension 4": (dim_4_test, dim_4_expected),
                 }
        for case, (test_data, expected) in cases.items():
            with self.subTest(case=case):
                test_t = encoder.transform(test_data)
                self.assertEqual(test_t.to_numpy().tolist(), expected)

    def test_sum_encoder_2cols(self):
        """Test the SumEncoder with two columns."""
        train = [['A', 'A'], ['B', 'B'], ['C', 'C']]

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        obtained = encoder.transform(train)

        expected = [
            a_encoding*2,
            b_encoding*2,
            c_encoding*2,
        ]
        self.assertEqual(obtained.to_numpy().tolist(), expected)

    def test_multiple_columns_correct_order(self):
        """Test that the order is correct when auto-detecting multiple columns."""
        train = pd.DataFrame(
            {
                'col1': [1, 2, 3, 4],
                'col2': ['A', 'B', 'C', 'D'],
                'col3': [1, 2, 3, 4],
                'col4': ['A', 'B', 'C', 'A'],
            },
            columns=['col1', 'col2', 'col3', 'col4'],
        )
        expected_columns = [
            'col1',
            'col2_0',
            'col2_1',
            'col2_2',
            'col3',
            'col4_0',
            'col4_1',
        ]
        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')

        encoder.fit(train)
        columns = encoder.transform(train).columns.to_numpy()

        self.assertTrue(np.array_equal(expected_columns, columns))

    def test_handle_missing_is_indicator(self):
        """Test that missing values are encoded with an indicator column."""
        with self.subTest("missing values in the training set are encoded with an "
                          "indicator column"):
            train = ['A', 'B', np.nan]

            encoder = encoders.SumEncoder(handle_missing='indicator', handle_unknown='value')
            result = encoder.fit_transform(train)

            expected = [a_encoding, b_encoding, c_encoding]
            self.assertListEqual(result.to_numpy().tolist(), expected)

        with self.subTest("should fit an indicator column for missing values "
                          "even if not present in the training set"):
            train = ['A', 'B']

            encoder = encoders.SumEncoder(handle_missing='indicator', handle_unknown='value')
            result = encoder.fit_transform(train)

            expected = [a_encoding, b_encoding]
            self.assertEqual(result.to_numpy().tolist(), expected)

            test = ['A', 'B', np.nan]
            result = encoder.transform(test)
            expected = [a_encoding, b_encoding, c_encoding]
            self.assertEqual(result.to_numpy().tolist(), expected)

            # unknown value should be encoded with value strategy, i.e. zeros for all columns
            test = ['A', 'B', 'C']
            result = encoder.transform(test)
            expected = [a_encoding, b_encoding, [0, 0]]
            self.assertEqual(result.to_numpy().tolist(), expected)

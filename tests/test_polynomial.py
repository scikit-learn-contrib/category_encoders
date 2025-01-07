"""Tests for the PolynomialEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd

from tests.helpers import deep_round

a_encoding = [-0.7071067811865476, 0.40824829046386313]
b_encoding = [-5.551115123125783e-17, -0.8164965809277261]
c_encoding = [0.7071067811865475, 0.4082482904638631]


class TestPolynomialEncoder(TestCase):
    """Tests for the PolynomialEncoder."""

    def test_handle_missing_and_unknown(self):
        """Test that missing and unknown values are treated as values."""
        train = ['A', 'B', 'C']
        expected_encoding_unknown = [0, 0]
        expected_1 = [a_encoding, expected_encoding_unknown, expected_encoding_unknown]
        expected_2 = [b_encoding, expected_encoding_unknown, expected_encoding_unknown]
        expected_3 = [a_encoding, b_encoding, c_encoding, expected_encoding_unknown]
        expected_4 = [expected_encoding_unknown, b_encoding, c_encoding, expected_encoding_unknown]
        cases = {"should preserve dimension 1": (['A', 'D', 'E'], expected_1),
                 "should preserve dimension 2": (['B', 'D', 'E'], expected_2),
                 "should preserve dimension 3": (['A', 'B', 'C', None], expected_3),
                 "should preserve dimension 4": (['D', 'B', 'C', None], expected_4),
                 }
        for case, (test_data, expected) in cases.items():
            with self.subTest(case=case):
                encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
                encoder.fit(train)
                test_t = encoder.transform(test_data)
                self.assertEqual(deep_round(test_t.to_numpy().tolist()), deep_round(expected))

    def test_polynomial_encoder_2cols(self):
        """Test the PolynomialEncoder with two columns."""
        train = [['A', 'A'], ['B', 'B'], ['C', 'C']]

        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        obtained = encoder.transform(train)

        expected = [
            a_encoding* 2,
            b_encoding* 2,
            c_encoding* 2,
        ]
        self.assertEqual(deep_round(obtained.to_numpy().tolist()), deep_round(expected))

    def test_correct_order(self):
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
        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')

        encoder.fit(train)
        columns = encoder.transform(train).columns.to_numpy()

        self.assertTrue(np.array_equal(expected_columns, columns))

    def test_handle_missing_is_indicator(self):
        """Test that missing values are encoded with an indicator column."""
        with self.subTest("missing values in the training set are encoded with an "
                          "indicator column"):
            train = ['A', 'B', np.nan]

            encoder = encoders.PolynomialEncoder(handle_missing='indicator', handle_unknown='value')
            result = encoder.fit_transform(train)

            expected = [a_encoding, b_encoding, c_encoding]
            self.assertListEqual(deep_round(result.to_numpy().tolist()), deep_round(expected))

        with self.subTest("should fit an indicator column for missing values "
                          "even if not present in the training set"):
            train = ['A', 'B']

            encoder = encoders.PolynomialEncoder(handle_missing='indicator', handle_unknown='value')
            result = encoder.fit_transform(train)

            expected = [a_encoding, b_encoding]
            self.assertEqual(deep_round(result.to_numpy().tolist()), deep_round(expected))

            test = ['A', 'B', np.nan]
            result = encoder.transform(test)
            expected = [a_encoding, b_encoding, c_encoding]
            self.assertEqual(deep_round(result.to_numpy().tolist()), deep_round(expected))

            # unknown value is encoded as zeros
            test = ['A', 'B', 'C']
            result = encoder.transform(test)
            expected = [a_encoding, b_encoding, [0, 0]]
            self.assertEqual(deep_round(result.to_numpy().tolist()), deep_round(expected))






import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import category_encoders as encoders
from category_encoders.tests.helpers import deep_round

a_encoding = [1, -0.7071067811865476, 0.40824829046386313]
b_encoding = [1, -5.551115123125783e-17, -0.8164965809277261]
c_encoding = [1, 0.7071067811865475, 0.4082482904638631]


class TestPolynomialEncoder(TestCase):

    def test_polynomial_encoder_preserve_dimension_1(self):
        train = ['A', 'B', 'C']
        test = ['A', 'D', 'E']

        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [a_encoding,
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(deep_round(test_t.values.tolist()), deep_round(expected))

    def test_polynomial_encoder_preserve_dimension_2(self):
        train = ['A', 'B', 'C']
        test = ['B', 'D', 'E']

        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [b_encoding,
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(deep_round(test_t.values.tolist()), deep_round(expected))

    def test_polynomial_encoder_preserve_dimension_3(self):
        train = ['A', 'B', 'C']
        test = ['A', 'B', 'C', None]

        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding,
                    [1, 0, 0]]
        self.assertEqual(deep_round(test_t.values.tolist()), deep_round(expected))

    def test_polynomial_encoder_preserve_dimension_4(self):
        train = ['A', 'B', 'C']
        test = ['D', 'B', 'C', None]

        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, 0, 0],
                    b_encoding,
                    c_encoding,
                    [1, 0, 0]]
        self.assertEqual(deep_round(test_t.values.tolist()), deep_round(expected))

    def test_polynomial_encoder_2cols(self):
        train = [['A', 'A'], ['B', 'B'], ['C', 'C']]

        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        obtained = encoder.transform(train)

        expected = [[1, a_encoding[1], a_encoding[2], a_encoding[1], a_encoding[2]],
                    [1, b_encoding[1], b_encoding[2], b_encoding[1], b_encoding[2]],
                    [1, c_encoding[1], c_encoding[2], c_encoding[1], c_encoding[2]]]
        self.assertEqual(deep_round(obtained.values.tolist()), deep_round(expected))

    def test_polynomial_encoder_2StringCols_ExpectCorrectOrder(self):
        train = pd.DataFrame({'col1': [1, 2, 3, 4],
                              'col2': ['A', 'B', 'C', 'D'],
                              'col3': [1, 2, 3, 4],
                              'col4': ['A', 'B', 'C', 'A']
                              },
                             columns=['col1', 'col2', 'col3', 'col4'])
        expected_columns = ['intercept', 'col1', 'col2_0', 'col2_1', 'col2_2', 'col3', 'col4_0', 'col4_1']
        encoder = encoders.PolynomialEncoder(handle_unknown='value', handle_missing='value')

        encoder.fit(train)
        columns = encoder.transform(train).columns.values

        self.assertItemsEqual(expected_columns, columns)

    def test_HandleMissingIndicator_NanInTrain_ExpectAsColumn(self):
        train = ['A', 'B', np.nan]

        encoder = encoders.PolynomialEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding]
        self.assertEqual(deep_round(result.values.tolist()), deep_round(expected))

    def test_HandleMissingIndicator_HaveNoNan_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.PolynomialEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [a_encoding,
                    b_encoding]
        self.assertEqual(deep_round(result.values.tolist()), deep_round(expected))

    def test_HandleMissingIndicator_NanNoNanInTrain_ExpectAsNanColumn(self):
        train = ['A', 'B']
        test = ['A', 'B', np.nan]

        encoder = encoders.PolynomialEncoder(handle_missing='indicator', handle_unknown='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding]
        self.assertEqual(deep_round(result.values.tolist()), deep_round(expected))

    def test_HandleUnknown_HaveNoUnknownInTrain_ExpectIndicatorInTest(self):
        train = ['A', 'B']
        test = ['A', 'B', 'C']

        encoder = encoders.PolynomialEncoder(handle_unknown='indicator')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding]
        self.assertEqual(deep_round(result.values.tolist()), deep_round(expected))

    def test_HandleUnknown_HaveOnlyKnown_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.PolynomialEncoder(handle_unknown='indicator')
        result = encoder.fit_transform(train)

        expected = [a_encoding,
                    b_encoding]
        self.assertEqual(deep_round(result.values.tolist()), deep_round(expected))

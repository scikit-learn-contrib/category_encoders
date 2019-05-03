import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import category_encoders as encoders

a_encoding = [1, 1, 0]
b_encoding = [1, 0, 1]
c_encoding = [1, -1, -1]


class TestSumEncoder(TestCase):

    def test_sum_encoder_preserve_dimension_1(self):
        train = ['A', 'B', 'C']
        test = ['A', 'D', 'E']

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [a_encoding,
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_sum_encoder_preserve_dimension_2(self):
        train = ['A', 'B', 'C']
        test = ['B', 'D', 'E']

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [b_encoding,
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_sum_encoder_preserve_dimension_3(self):
        train = ['A', 'B', 'C']
        test = ['A', 'B', 'C', None]

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding,
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_sum_encoder_preserve_dimension_4(self):
        train = ['A', 'B', 'C']
        test = ['D', 'B', 'C', None]

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, 0, 0],
                    b_encoding,
                    c_encoding,
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_sum_encoder_2cols(self):
        train = [['A', 'A'], ['B', 'B'], ['C', 'C']]

        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        obtained = encoder.transform(train)

        expected = [[1, a_encoding[1], a_encoding[2], a_encoding[1], a_encoding[2]],
                    [1, b_encoding[1], b_encoding[2], b_encoding[1], b_encoding[2]],
                    [1, c_encoding[1], c_encoding[2], c_encoding[1], c_encoding[2]]]
        self.assertEqual(obtained.values.tolist(), expected)

    def test_sum_encoder_2StringCols_ExpectCorrectOrder(self):
        train = pd.DataFrame({'col1': [1, 2, 3, 4],
                              'col2': ['A', 'B', 'C', 'D'],
                              'col3': [1, 2, 3, 4],
                              'col4': ['A', 'B', 'C', 'A']
                              },
                             columns=['col1', 'col2', 'col3', 'col4'])
        expected_columns = ['intercept', 'col1', 'col2_0', 'col2_1', 'col2_2', 'col3', 'col4_0', 'col4_1']
        encoder = encoders.SumEncoder(handle_unknown='value', handle_missing='value')

        encoder.fit(train)
        columns = encoder.transform(train).columns.values

        self.assertItemsEqual(expected_columns, columns)

    def test_HandleMissingIndicator_NanInTrain_ExpectAsColumn(self):
        train = ['A', 'B', np.nan]

        encoder = encoders.SumEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_HaveNoNan_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.SumEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [a_encoding,
                    b_encoding]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_NanNoNanInTrain_ExpectAsNanColumn(self):
        train = ['A', 'B']
        test = ['A', 'B', np.nan]

        encoder = encoders.SumEncoder(handle_missing='indicator', handle_unknown='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveNoUnknownInTrain_ExpectIndicatorInTest(self):
        train = ['A', 'B']
        test = ['A', 'B', 'C']

        encoder = encoders.SumEncoder(handle_unknown='indicator', handle_missing='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [a_encoding,
                    b_encoding,
                    c_encoding]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveOnlyKnown_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.SumEncoder(handle_unknown='indicator', handle_missing='value')
        result = encoder.fit_transform(train)

        expected = [a_encoding,
                    b_encoding]
        self.assertEqual(result.values.tolist(), expected)

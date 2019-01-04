import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import category_encoders as encoders


class TestBackwardsEncoder(TestCase):

    def test_backwards_difference_encoder_preserve_dimension_1(self):
        train = ['A', 'B', 'C']
        test = ['A', 'D', 'E']

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_backwards_difference_encoder_preserve_dimension_2(self):
        train = ['A', 'B', 'C']
        test = ['B', 'D', 'E']

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, 1 / 3.0, -1 / 3.0],
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_backwards_difference_encoder_preserve_dimension_3(self):
        train = ['A', 'B', 'C']
        test = ['A', 'B', 'C', None]

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, 2 / 3.0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_backwards_difference_encoder_preserve_dimension_4(self):
        train = ['A', 'B', 'C']
        test = ['D', 'B', 'C', None]

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, 0, 0],
                    [1, 1 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, 2 / 3.0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_backwards_difference_encoder_2cols(self):
        train = [['A', 'A'], ['B', 'B'], ['C', 'C']]

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        obtained = encoder.transform(train)

        expected = [[1, -2 / 3.0, -1 / 3.0, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0, 1 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, 2 / 3.0, 1 / 3.0, 2 / 3.0]]
        self.assertEqual(obtained.values.tolist(), expected)

    def test_backwards_difference_encoder_2StringCols_ExpectCorrectOrder(self):
        train = pd.DataFrame({'col1': [1, 2, 3, 4],
                              'col2': ['A', 'B', 'C', 'D'],
                              'col3': [1, 2, 3, 4],
                              'col4': ['A', 'B', 'C', 'A']
                              },
                             columns=['col1', 'col2', 'col3', 'col4'])
        expected_columns = ['intercept', 'col1', 'col2_0', 'col2_1', 'col2_2', 'col3', 'col4_0', 'col4_1']
        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='value', handle_missing='value')

        encoder.fit(train)
        columns = encoder.transform(train).columns.values

        self.assertItemsEqual(expected_columns, columns)

    def test_HandleMissingIndicator_NanInTrain_ExpectAsColumn(self):
        train = ['A', 'B', np.nan]

        encoder = encoders.BackwardDifferenceEncoder(handle_missing='indicator')
        result = encoder.fit_transform(train)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, 2 / 3.0]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_HaveNoNan_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.BackwardDifferenceEncoder(handle_missing='indicator')
        result = encoder.fit_transform(train)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_NanNoNanInTrain_ExpectAsNanColumn(self):
        train = ['A', 'B']
        test = ['A', 'B', np.nan]

        encoder = encoders.BackwardDifferenceEncoder(handle_missing='indicator')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, 2 / 3.0]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveNoUnknownInTrain_ExpectIndicatorInTest(self):
        train = ['A', 'B']
        test = ['A', 'B', 'C']

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='indicator')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, 2 / 3.0]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveOnlyKnown_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.BackwardDifferenceEncoder(handle_unknown='indicator')
        result = encoder.fit_transform(train)

        expected = [[1, -2 / 3.0, -1 / 3.0],
                    [1, 1 / 3.0, -1 / 3.0]]
        self.assertEqual(result.values.tolist(), expected)

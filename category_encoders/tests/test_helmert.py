import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np
import category_encoders as encoders


class TestHelmertEncoder(TestCase):

    def test_helmert_preserve_dimension_1(self):
        train = ['A', 'B', 'C']
        test = ['A', 'D', 'E']

        encoder = encoders.HelmertEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, -1, -1],
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_helmert_preserve_dimension_2(self):
        train = ['A', 'B', 'C']
        test = ['B', 'D', 'E']

        encoder = encoders.HelmertEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, 1, -1],
                    [1, 0, 0],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_helmert_preserve_dimension_3(self):
        train = ['A', 'B', 'C']
        test = ['A', 'B', 'C', None]

        encoder = encoders.HelmertEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, -1, -1],
                    [1, 1, -1],
                    [1, 0, 2],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_helmert_preserve_dimension_4(self):
        train = ['A', 'B', 'C']
        test = ['D', 'B', 'C', None]

        encoder = encoders.HelmertEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        test_t = encoder.transform(test)

        expected = [[1, 0, 0],
                    [1, 1, -1],
                    [1, 0, 2],
                    [1, 0, 0]]
        self.assertEqual(test_t.values.tolist(), expected)

    def test_helmert_2cols(self):
        train = [['A', 'A'], ['B', 'B'], ['C', 'C']]

        encoder = encoders.HelmertEncoder(handle_unknown='value', handle_missing='value')
        encoder.fit(train)
        obtained = encoder.transform(train)

        expected = [[1, -1, -1, -1, -1],
                    [1,  1, -1,  1, -1],
                    [1,  0,  2,  0,  2]]
        self.assertEqual(obtained.values.tolist(), expected)

    def test_helmert_2StringCols_ExpectCorrectOrder(self):
        train = pd.DataFrame({'col1': [1, 2, 3, 4],
                              'col2': ['A', 'B', 'C', 'D'],
                              'col3': [1, 2, 3, 4],
                              'col4': ['A', 'B', 'C', 'A']
                              },
                             columns=['col1', 'col2', 'col3', 'col4'])
        expected_columns = ['intercept', 'col1', 'col2_0', 'col2_1', 'col2_2', 'col3', 'col4_0', 'col4_1']
        encoder = encoders.HelmertEncoder(handle_unknown='value', handle_missing='value')

        encoder.fit(train)
        columns = encoder.transform(train).columns.values

        self.assertItemsEqual(expected_columns, columns)

    def test_HandleMissingIndicator_NanInTrain_ExpectAsColumn(self):
        train = ['A', 'B', np.nan]

        encoder = encoders.HelmertEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [[1, -1, -1],
                    [1, 1, -1],
                    [1, 0, 2]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_HaveNoNan_ExpectSecondColumn(self):
        train = ['A', 'B']

        encoder = encoders.HelmertEncoder(handle_missing='indicator', handle_unknown='value')
        result = encoder.fit_transform(train)

        expected = [[1, -1, -1],
                    [1, 1, -1]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleMissingIndicator_NanNoNanInTrain_ExpectAsNanColumn(self):
        train = ['A', 'B']
        test = ['A', 'B', np.nan]

        encoder = encoders.HelmertEncoder(handle_missing='indicator', handle_unknown='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [[1, -1, -1],
                    [1, 1, -1],
                    [1, 0, 2]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveNoUnknownInTrain_ExpectIndicatorInTest(self):
        train = ['A', 'B']
        test = ['A', 'B', 'C']

        encoder = encoders.HelmertEncoder(handle_unknown='indicator', handle_missing='value')
        encoder.fit(train)
        result = encoder.transform(test)

        expected = [[1, -1, -1],
                    [1, 1, -1],
                    [1, 0, 2]]
        self.assertEqual(result.values.tolist(), expected)

    def test_HandleUnknown_HaveOnlyKnown_ExpectExtraColumn(self):
        train = ['A', 'B']

        encoder = encoders.HelmertEncoder(handle_unknown='indicator', handle_missing='value')
        result = encoder.fit_transform(train)

        expected = [[1, -1, -1],
                    [1, 1, -1]]
        self.assertEqual(result.values.tolist(), expected)

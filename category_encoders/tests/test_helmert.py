import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+

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

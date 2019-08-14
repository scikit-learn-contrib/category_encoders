from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders as encoders


class TestMEstimateEncoder(TestCase):

    def test_reference_m0(self):
        x = ['A', 'A', 'B', 'B']
        y = [1, 1, 0, 1]
        x_t = ['A', 'B', 'C']

        encoder = encoders.MEstimateEncoder(m=0, handle_unknown='value', handle_missing='value')
        encoder.fit(x, y)
        scored = encoder.transform(x_t)

        expected = [[1],
                    [0.5],
                    [3./4.]]  # The prior probability
        self.assertEqual(scored.values.tolist(), expected)

    def test_reference_m1(self):
        x = ['A', 'A', 'B', 'B']
        y = [1, 1, 0, 1]
        x_t = ['A', 'B', 'C']

        encoder = encoders.MEstimateEncoder(m=1, handle_unknown='value', handle_missing='value')
        encoder.fit(x, y)
        scored = encoder.transform(x_t)

        expected = [[(2+3./4.)/(2+1)],
                    [(1+3./4.)/(2+1)],
                    [3./4.]]  # The prior probability
        self.assertEqual(scored.values.tolist(), expected)

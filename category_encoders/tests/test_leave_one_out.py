import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.test_utils as tu
import numpy as np

import category_encoders as encoders


np_X = tu.create_array(n_rows=100)
np_X_t = tu.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = tu.create_dataset(n_rows=100)
X_t = tu.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestLeaveOneOutEncoder(TestCase):

    def test_leave_one_out(self):
        enc = encoders.LeaveOneOutEncoder(verbose=1, sigma=0.1)
        enc.fit(X, y)
        tu.verify_numeric(enc.transform(X_t))
        tu.verify_numeric(enc.transform(X_t, y_t))

    def test_leave_one_out_values(self):
        df = pd.DataFrame({
            'color': ["a", "a", "a", "b", "b", "b"],
            'outcome': [1, 0, 0, 1, 0, 1]})

        X = df.drop('outcome', axis=1)
        y = df.drop('color', axis=1)

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'])
        obtained = ce_leave.fit_transform(X, y['outcome'])

        self.assertEqual([0.0, 0.5, 0.5, 0.5, 1.0, 0.5], list(obtained['color']))

    def test_leave_one_out_fit_callTwiceOnDifferentData_ExpectRefit(self):
        x_a = pd.DataFrame(data=['1', '2', '2', '2', '2', '2'], columns=['col_a'])
        x_b = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_b'])  # different values and name
        y_dummy = [True, False, True, False, True, False]
        encoder = encoders.LeaveOneOutEncoder()
        encoder.fit(x_a, y_dummy)
        encoder.fit(x_b, y_dummy)
        mapping = encoder.mapping
        self.assertEqual(1, len(mapping))
        self.assertIn('col_b', mapping)     # the model should have the updated mapping
        expected = pd.DataFrame({'sum': [2.0, 1.0], 'count': [3, 3]}, index=['1', '2'])
        pd.testing.assert_frame_equal(expected, mapping['col_b'], check_like=True)

    def test_leave_one_out_unique(self):
        X = pd.DataFrame(data=['1', '2', '2', '2', '3'], columns=['col'])
        y = np.array([1, 0, 1, 0, 1])

        encoder = encoders.LeaveOneOutEncoder(handle_unknown='value')
        result = encoder.fit(X, y).transform(X, y)

        self.assertFalse(result.isnull().any().any(), 'There should not be any missing value')
        expected = pd.DataFrame(data=[y.mean(), 0.5, 0, 0.5, y.mean()], columns=['col'])
        pd.testing.assert_frame_equal(expected, result)

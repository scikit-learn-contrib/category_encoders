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
        enc = encoders.LeaveOneOutEncoder(verbose=1, randomized=True, sigma=0.1)
        enc.fit(X, y)
        tu.verify_numeric(enc.transform(X_t))
        tu.verify_numeric(enc.transform(X_t, y_t))

    def test_leave_one_out_values(self):
        df = pd.DataFrame({
            'color': ["a", "a", "a", "b", "b", "b"],
            'outcome': [1, 0, 0, 1, 0, 1]})

        X = df.drop('outcome', axis=1)
        y = df.drop('color', axis=1)

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], randomized=False)
        obtained = ce_leave.fit_transform(X, y['outcome'])

        self.assertEquals([0.0, 0.5, 0.5, 0.5, 1.0, 0.5], list(obtained['color']))

    def test_leave_one_out_fit_callTwiceOnDifferentData_ExpectRefit(self):
        x_a = pd.DataFrame(data=['1', '2', '2', '2', '2', '2'], columns=['col_a'])
        x_b = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_b'])  # different values and name
        y_dummy = [True, False, True, False, True, False]
        encoder = encoders.LeaveOneOutEncoder()
        encoder.fit(x_a, y_dummy)
        encoder.fit(x_b, y_dummy)
        mapping = encoder.mapping
        self.assertEqual(1, len(mapping))
        col_b_mapping = mapping[0]
        self.assertEqual('col_b', col_b_mapping['col']) # the model must get updated
        self.assertEqual({'sum': 2.0, 'count': 3, 'mean': 2.0/3.0}, col_b_mapping['mapping']['1'])
        self.assertEqual({'sum': 1.0, 'count': 3, 'mean': 01.0/3.0}, col_b_mapping['mapping']['2'])

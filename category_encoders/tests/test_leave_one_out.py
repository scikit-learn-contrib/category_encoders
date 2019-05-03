import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np

import category_encoders as encoders


np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestLeaveOneOutEncoder(TestCase):

    def test_leave_one_out(self):
        enc = encoders.LeaveOneOutEncoder(verbose=1, sigma=0.1)
        enc.fit(X, y)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t))

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
        expected = pd.DataFrame({'sum': [2.0, 1.0], 'count': [3, 3]}, index=['1', '2'],  columns=['sum', 'count'])
        np.testing.assert_equal(expected.values, mapping['col_b'].values)

    def test_leave_one_out_unique(self):
        X = pd.DataFrame(data=['1', '2', '2', '2', '3'], columns=['col'])
        y = np.array([1, 0, 1, 0, 1])

        encoder = encoders.LeaveOneOutEncoder(handle_unknown='value')
        result = encoder.fit(X, y).transform(X, y)

        self.assertFalse(result.isnull().any().any(), 'There should not be any missing value')
        expected = pd.DataFrame(data=[y.mean(), 0.5, 0, 0.5, y.mean()], columns=['col'])
        pd.testing.assert_frame_equal(expected, result)

    def test_HandleMissingIsValueAndNanInTrain_ExpectAtValueSet(self):
        df = pd.DataFrame({
            'color': [np.nan, np.nan, np.nan, "b", "b", "b"],
            'outcome': [2, 2, 0, 1, 0, 1]})

        X = df.drop('outcome', axis=1)
        y = df.drop('color', axis=1)

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_missing='value')
        obtained = ce_leave.fit_transform(X, y['outcome'])

        self.assertEqual([1, 1, 2, 0.5, 1.0, 0.5], list(obtained['color']))

    def test_HandleMissingIsValueAndNanInTest_ExpectMean(self):
        df = pd.DataFrame({
            'color': ["a", "a", "a", "b", "b", "b"],
            'outcome': [1.6, 0, 0, 1, 0, 1]})

        train = df.drop('outcome', axis=1)
        target = df.drop('color', axis=1)
        test = pd.Series([np.nan, 'b'], name='color')
        test_target = pd.Series([0, 0])

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_missing='value')
        ce_leave.fit(train, target['outcome'])
        obtained = ce_leave.transform(test, test_target)

        self.assertEqual([.6, 1.0], list(obtained['color']))

    def test_HandleMissingIsValueAndNanInTestAndNoTestTarget_ExpectMean(self):
        df = pd.DataFrame({
            'color': ["a", "a", "a", "b", "b", "b"],
            'outcome': [1, 0, 0, 1, 0, 1]})

        train = df.drop('outcome', axis=1)
        target = df.drop('color', axis=1)
        test = pd.Series([np.nan, 'b'], name='color')

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_missing='value')
        ce_leave.fit(train, target['outcome'])
        obtained = ce_leave.transform(test)

        self.assertEqual([.5, 2/3.0], list(obtained['color']))

    def test_HandleUnknownValue_HaveUnknownInTest_ExpectMean(self):
        train = pd.Series(["a", "a", "a", "b", "b", "b"], name='color')
        target = pd.Series([1.6, 0, 0, 1, 0, 1], name='target')
        test = pd.Series(['b', 'c'], name='color')
        test_target = pd.Series([0, 0])

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_unknown='value')
        ce_leave.fit(train, target)
        obtained = ce_leave.transform(test, test_target)

        self.assertEqual([1.0, .6], list(obtained['color']))

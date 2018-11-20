import doctest
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import sklearn
import category_encoders.tests.test_utils as tu
from sklearn.utils.estimator_checks import check_transformer_general, check_transformers_unfitted
from unittest2 import TestSuite, TextTestRunner, TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders

__author__ = 'willmcginnis'


# data definitions
np_X = tu.create_array(n_rows=100)
np_X_t = tu.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = tu.create_dataset(n_rows=100)
X_t = tu.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


# this class utilises parametrised tests where we loop over different encoders
class TestEncoders(TestCase):

    def test_np(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):

                # Encode a numpy array
                enc = getattr(encoders, encoder_name)()
                enc.fit(np_X, np_y)
                tu.verify_numeric(enc.transform(np_X_t))

    def test_classification(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical']

                enc = getattr(encoders, encoder_name)(cols=cols)
                enc.fit(X, np_y)
                tu.verify_numeric(enc.transform(X_t))

                enc = getattr(encoders, encoder_name)(verbose=1)
                enc.fit(X, np_y)
                tu.verify_numeric(enc.transform(X_t))

                enc = getattr(encoders, encoder_name)(drop_invariant=True)
                enc.fit(X, np_y)
                tu.verify_numeric(enc.transform(X_t))

                enc = getattr(encoders, encoder_name)(return_df=False)
                enc.fit(X, np_y)
                self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))
                self.assertEqual(enc.transform(X_t).shape[0], X_t.shape[0], 'Row count must not change')

                # documented in issue #122
                # when we use the same encoder on two different datasets, it should not explode
                # X_a = pd.DataFrame(data=['1', '2', '2', '2', '2', '2'], columns=['col_a'])
                # X_b = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_b']) # different values and name
                # y_dummy = [True, False, True, False, True, False]
                # enc = getattr(encoders, encoder_name)()
                # enc.fit(X_a, y_dummy)
                # enc.fit(X_b, y_dummy)
                # verify_numeric(enc.transform(X_b))

    def test_impact_encoders(self):
        for encoder_name in ['LeaveOneOutEncoder', 'TargetEncoder', 'WOEEncoder']:
            with self.subTest(encoder_name=encoder_name):

                # encode a numpy array and transform with the help of the target
                enc = getattr(encoders, encoder_name)()
                enc.fit(np_X, np_y)
                tu.verify_numeric(enc.transform(np_X_t, np_y_t))

                # target is a DataFrame
                enc = getattr(encoders, encoder_name)()
                enc.fit(X, y)
                tu.verify_numeric(enc.transform(X_t, y_t))

                # when we run transform(X, y) and there is a new value in X, something is wrong and we raise an error
                enc = getattr(encoders, encoder_name)(impute_missing=True, handle_unknown='error', cols=['extra'])
                enc.fit(X, y)
                self.assertRaises(ValueError, enc.transform, (X_t, y_t))

    def test_error_handling(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):

                # we exclude some columns
                X = tu.create_dataset(n_rows=100)
                X = X.drop(['unique_str', 'none'], axis=1)
                X_t = tu.create_dataset(n_rows=50, extras=True)
                X_t = X_t.drop(['unique_str', 'none'], axis=1)

                # illegal state, we have to first train the encoder...
                enc = getattr(encoders, encoder_name)()
                with self.assertRaises(ValueError):
                    enc.transform(X)

                # wrong count of attributes
                enc = getattr(encoders, encoder_name)()
                enc.fit(X, y)
                with self.assertRaises(ValueError):
                    enc.transform(X_t.iloc[:, 0:3])

                # no cols
                enc = getattr(encoders, encoder_name)(cols=[])
                enc.fit(X, y)
                self.assertTrue(enc.transform(X_t).equals(X_t))

    def test_handle_unknown_error(self):
        # BaseN has problems with None -> ignore None
        X = tu.create_dataset(n_rows=100, has_none=False)
        X_t = tu.create_dataset(n_rows=50, extras=True, has_none=False)

        for encoder_name in (set(encoders.__all__) - {'HashingEncoder'}):  # HashingEncoder supports new values by design -> excluded
            with self.subTest(encoder_name=encoder_name):

                # new value during scoring
                enc = getattr(encoders, encoder_name)(handle_unknown='error')
                enc.fit(X, y)
                with self.assertRaises(ValueError):
                    _ = enc.transform(X_t)

    def test_sklearn_compliance(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):

                # in sklearn < 0.19.0, these methods require classes,
                # in sklearn >= 0.19.0, these methods require instances
                if sklearn.__version__ < '0.19.0':
                    encoder = getattr(encoders, encoder_name)
                else:
                    encoder = getattr(encoders, encoder_name)()

                check_transformer_general(encoder_name, encoder)
                check_transformers_unfitted(encoder_name, encoder)

    def test_inverse_transform(self):
        # we do not allow None in these data (but "none" column without any None is ok)
        X = tu.create_dataset(n_rows=100, has_none=False)
        X_t = tu.create_dataset(n_rows=50, has_none=False)
        X_t_extra = tu.create_dataset(n_rows=50, extras=True, has_none=False)
        cols = ['underscore', 'none', 'extra', 321, 'categorical']

        for encoder_name in ['BaseNEncoder', 'BinaryEncoder', 'OneHotEncoder', 'OrdinalEncoder']:
            with self.subTest(encoder_name=encoder_name):

                # simple run
                enc = getattr(encoders, encoder_name)(verbose=1, cols=cols)
                enc.fit(X)
                tu.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))

                # when a new value is encountered, do not raise an exception
                enc = getattr(encoders, encoder_name)(verbose=1, cols=cols)
                enc.fit(X, y)
                _ = enc.inverse_transform(enc.transform(X_t_extra))

    def test_types(self):
        X = pd.DataFrame({
            'Int': [1, 2, 1, 2],
            'Float': [1.1, 2.2, 3.3, 4.4],
            'Complex': [3.45J, 3.45J, 3.45J, 3.45J],
            'None': [None, None, None, None],
            'Str': ['a', 'c', 'c', 'd'],
            'PdTimestamp': [pd.Timestamp('2012-05-01'), pd.Timestamp('2012-05-02'), pd.Timestamp('2012-05-03'), pd.Timestamp('2012-05-06')],
            'PdTimedelta': [pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('1 days'), pd.Timedelta('1 days')],
            'TimeDelta': [timedelta(-9999), timedelta(-9), timedelta(-1), timedelta(999)],
            'Bool': [False, True, True, False],
            'Tuple': [('a', 'tuple'), ('a', 'tuple'), ('a', 'tuple'), ('b', 'tuple')],
            'Categorical': pd.Categorical(list('bbea'), categories=['e', 'a', 'b'], ordered=True),
            # 'List': [[1,2], [2,3], [3,4], [4,5]],
            # 'Dictionary': [{1: "a", 2: "b"}, {1: "a", 2: "b"}, {1: "a", 2: "b"}, {1: "a", 2: "b"}],
            # 'Set': [{'John', 'Jane'}, {'John', 'Jane'}, {'John', 'Jane'}, {'John', 'Jane'}],
            # 'Array': [array('i'), array('i'), array('i'), array('i')]
        })
        y = [1, 0, 0, 1]

        for encoder_name in encoders.__all__:
            encoder = getattr(encoders, encoder_name)()
            encoder.fit_transform(X, y)

    def test_preserve_column_order(self):
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]}, columns=['Trend', 'target'])

        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                print(encoder_name)
                encoder = getattr(encoders, encoder_name)()
                result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
                columns = result.columns.values

                self.assertTrue('target' in columns[-1], "Target must be the last column as in the input")

    def test_tmp_column_name(self):
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT'],
             'Trend_tmp': ['UP', 'UP', 'DOWN', 'FLAT'],
             'target': [1, 1, 0, 0]}, columns=['Trend', 'Trend_tmp', 'target'])

        for encoder_name in ['LeaveOneOutEncoder', 'TargetEncoder', 'WOEEncoder']:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)()
                _ = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])

    def test_preserve_names(self):
        binary_cat_example = pd.DataFrame(
            {'ignore': ['UP', 'UP', 'DOWN', 'FLAT'],
             'feature': ['UP', 'UP', 'DOWN', 'FLAT'],
             'target': [1, 1, 0, 0]}, columns=['ignore', 'feature', 'target'])

        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)(cols=['feature'])
                result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
                columns = result.columns.values

                self.assertTrue('ignore' in columns, "Column 'ignore' is missing in: " + str(columns))

    def test_unique_column_is_not_predictive(self):
        for encoder_name in ['LeaveOneOutEncoder', 'TargetEncoder', 'WOEEncoder']:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)()
                result = encoder.fit_transform(X[['unique_str']], y)
                self.assertTrue(all(result.var() < 0.001), 'The unique string column must not be predictive of the label')

    # beware: for some reason doctest does not raise exceptions - you have to read the text output
    def test_doc(self):
        suite = TestSuite()

        for filename in os.listdir('../'):
            if filename.endswith(".py"):
                suite.addTest(doctest.DocFileSuite('../' + filename))

        runner = TextTestRunner(verbosity=2)
        runner.run(suite)

    def test_cols(self):
        # Test cols argument with different data types, which are array-like or scalars
        cols_list = ['extra', 'invariant']
        cols_types = [cols_list, pd.Series(cols_list), np.array(cols_list), 'extra', 321, set(cols_list),
                      ('extra', 'invariant'), pd.Categorical(cols_list, categories=cols_list)]

        for encoder_name in encoders.__all__:
            for cols in cols_types:
                with self.subTest(encoder_name=encoder_name, cols=cols):
                    enc = getattr(encoders, encoder_name)(cols=cols)
                    enc.fit(X, y)
                    enc.transform(X_t)

    def test_noncontiguous_index(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):

                enc = getattr(encoders, encoder_name)(cols=['x'])
                data = pd.DataFrame({'x': ['a', 'b', np.nan, 'd', 'e'], 'y': [1, 0, 1, 0, 1]}).dropna()
                _ = enc.fit_transform(data[['x']], data['y'])

    def test_duplicate_index_value(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(cols=['x'])
                data = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e'], 'y': [1, 0, 1, 0, 1]}, index=[1, 2, 2, 3, 4])
                result = enc.fit_transform(data[['x']], data['y'])
                self.assertEqual(5, len(result))

    def test_string_index(self):
        # https://github.com/scikit-learn-contrib/categorical-encoding/issues/131

        bunch = sklearn.datasets.load_boston()
        y = (bunch.target > 20)
        X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        X.index = X.index.values.astype(str)

        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(cols=['CHAS', 'RAD'])
                result = enc.fit_transform(X, y)
                self.assertFalse(result.isnull().values.any(), 'There should not be any missing value!')

    def test_get_feature_names(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                # These 3 need y also
                if not encoder_name in ['TargetEncoder','WOEEncoder','LeaveOneOutEncoder']:
                    obtained = enc.fit(X).get_feature_names()
                    expected = enc.transform(X).columns.tolist()
                else:
                    obtained = enc.fit(X, y).get_feature_names()
                    expected = enc.transform(X, y).columns.tolist()
                self.assertEqual(obtained, expected)

    def test_get_feature_names_drop_invariant(self):
        # TODO: What could a DF look like that results in constant
        # columns for all encoders?
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(drop_invariant=True)
                # These 3 need y also
                if not encoder_name in ['TargetEncoder','WOEEncoder','LeaveOneOutEncoder']:
                    obtained = enc.fit(X).get_feature_names()
                    expected = enc.transform(X).columns.tolist()
                else:
                    obtained = enc.fit(X, y).get_feature_names()
                    expected = enc.transform(X, y).columns.tolist()
                self.assertEqual(obtained, expected)

    def test_get_feature_names_not_set(self):
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                self.assertRaises(ValueError, enc.get_feature_names)

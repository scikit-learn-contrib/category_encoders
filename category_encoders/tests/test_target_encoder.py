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


class TestTargetEncoder(TestCase):

    def test_target_encoder(self):

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2)
        enc.fit(X, y)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t))

    def test_target_encoder_fit_HaveConstructorSetSmoothingAndMinSamplesLeaf_ExpectUsedInFit(self):
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        encoder.fit(binary_cat_example, binary_cat_example['target'])
        trend_mapping = encoder.mapping['Trend']
        ordinal_mapping = encoder.ordinal_encoder.category_mapping[0]['mapping']

        self.assertAlmostEqual(0.4125, trend_mapping[ordinal_mapping.loc['DOWN']], delta=1e-4)
        self.assertEqual(0.5, trend_mapping[ordinal_mapping.loc['FLAT']])
        self.assertAlmostEqual(0.5874, trend_mapping[ordinal_mapping.loc['UP']], delta=1e-4)

    def test_target_encoder_fit_transform_HaveConstructorSetSmoothingAndMinSamplesLeaf_ExpectCorrectValueInResult(self):
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
        values = result['Trend'].values
        self.assertAlmostEqual(0.5874, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5874, values[1], delta=1e-4)
        self.assertAlmostEqual(0.4125, values[2], delta=1e-4)
        self.assertEqual(0.5, values[3])

    def test_target_encoder_fit_transform_HaveCategoricalColumn_ExpectCorrectValueInResult(self):
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': pd.Categorical(['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
                                     categories=['UP', 'FLAT', 'DOWN']),
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
        values = result['Trend'].values
        self.assertAlmostEqual(0.5874, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5874, values[1], delta=1e-4)
        self.assertAlmostEqual(0.4125, values[2], delta=1e-4)
        self.assertEqual(0.5, values[3])

    def test_target_encoder_fit_transform_HaveNanValue_ExpectCorrectValueInResult(self):
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': pd.Series([np.nan, np.nan, 'DOWN', 'FLAT', 'DOWN', np.nan, 'DOWN', 'FLAT', 'FLAT', 'FLAT']),
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
        values = result['Trend'].values
        self.assertAlmostEqual(0.5874, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5874, values[1], delta=1e-4)
        self.assertAlmostEqual(0.4125, values[2], delta=1e-4)
        self.assertEqual(0.5, values[3])

    def test_target_encoder_noncontiguous_index(self):
        data = pd.DataFrame({'x': ['a', 'b', np.nan, 'd', 'e'], 'y': range(5)}).dropna()
        result = encoders.TargetEncoder(cols=['x']).fit_transform(data[['x']], data['y'])
        self.assertTrue(np.allclose(result, 2.0))

    def test_HandleMissingIsValueAndNanInTest_ExpectMean(self):
        df = pd.DataFrame({
            'color': ["a", "a", "a", "b", "b", "b"],
            'outcome': [1.6, 0, 0, 1, 0, 1]})

        train = df.drop('outcome', axis=1)
        target = df.drop('color', axis=1)
        test = pd.Series([np.nan, 'b'], name='color')
        test_target = pd.Series([0, 0])

        enc = encoders.TargetEncoder(cols=['color'], handle_missing='value')
        enc.fit(train, target['outcome'])
        obtained = enc.transform(test, test_target)

        self.assertEqual(.6, list(obtained['color'])[0])

    def test_HandleUnknownValue_HaveUnknownInTest_ExpectMean(self):
        train = pd.Series(["a", "a", "a", "b", "b", "b"], name='color')
        target = pd.Series([1.6, 0, 0, 1, 0, 1], name='target')
        test = pd.Series(['c', 'b'], name='color')
        test_target = pd.Series([0, 0])

        enc = encoders.TargetEncoder(cols=['color'], handle_unknown='value')
        enc.fit(train, target)
        obtained = enc.transform(test, test_target)

        self.assertEqual(.6, list(obtained['color'])[0])

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


class TestTargetEncoder(TestCase):

    def test_target_encoder(self):

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2)
        enc.fit(X, y)
        tu.verify_numeric(enc.transform(X_t))
        tu.verify_numeric(enc.transform(X_t, y_t))

    def test_target_encoder_fit_HaveConstructorSetSmoothingAndMinSamplesLeaf_ExpectUsedInFit(self):
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        encoder.fit(binary_cat_example, binary_cat_example['target'])
        trend_mapping = encoder.mapping['Trend']
        self.assertAlmostEqual(0.4125, trend_mapping['DOWN'], delta=1e-4)
        self.assertEqual(0.5, trend_mapping['FLAT'])
        self.assertAlmostEqual(0.5874, trend_mapping['UP'], delta=1e-4)

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

    def test_target_encoder_noncontiguous_index(self):
        data = pd.DataFrame({'x': ['a', 'b', np.nan, 'd', 'e'], 'y': range(5)}).dropna()
        result = encoders.TargetEncoder(cols=['x']).fit_transform(data[['x']], data['y'])
        self.assertTrue(np.allclose(result, 2.0))


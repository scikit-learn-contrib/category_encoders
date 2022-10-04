import pandas as pd
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+
import tests.helpers as th
import numpy as np
from category_encoders.datasets import load_postcodes, load_compass

import category_encoders as encoders


class TestTargetEncoder(TestCase):

    def setUp(self):
        self.hierarchical_cat_example = pd.DataFrame(
            {
                'Compass': ['N', 'N', 'NE', 'NE', 'NE', 'SE', 'SE', 'S', 'S', 'S', 'S', 'W', 'W', 'W', 'W', 'W'],
                'Speed': ['slow', 'slow', 'slow', 'slow', 'medium', 'medium', 'medium', 'fast', 'fast', 'fast', 'fast',
                          'fast', 'fast', 'fast', 'fast', 'fast'],
                'Animal': ['Cat', 'Cat', 'Cat', 'Cat', 'Cat', 'Dog', 'Dog', 'Dog', 'Dog',
                           'Dog', 'Dog', 'Tiger', 'Tiger', 'Wolf', 'Wolf', 'Cougar'],
                'Plant': ['Rose', 'Rose', 'Rose', 'Rose', 'Daisy', 'Daisy', 'Daisy', 'Daisy', 'Daffodil',
                          'Daffodil', 'Daffodil', 'Daffodil', 'Bluebell', 'Bluebell', 'Bluebell', 'Bluebell'],
                'target': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
            }, columns=['Compass', 'Speed', 'Animal', 'Plant', 'target'])
        self.hierarchical_map = {
            'Compass': {
                'N': ('N', 'NE'),
                'S': ('S', 'SE'),
                'W': 'W'
            },
            'Animal': {
                'Feline': ('Cat', 'Tiger', 'Cougar'),
                'Canine': ('Dog', 'Wolf')
            },
            'Plant': {
                'Flower': ('Rose', 'Daisy', 'Daffodil', 'Bluebell'),
                'Tree': ('Ash', 'Birch')
            },
        }

    def test_target_encoder(self):
        np_X = th.create_array(n_rows=100)
        np_X_t = th.create_array(n_rows=50, extras=True)
        np_y = np.random.randn(np_X.shape[0]) > 0.5
        np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
        X = th.create_dataset(n_rows=100)
        X_t = th.create_dataset(n_rows=50, extras=True)
        y = pd.DataFrame(np_y)
        y_t = pd.DataFrame(np_y_t)
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

    def test_hierarchical_smoothing(self):

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=self.hierarchical_map, cols=['Compass'])
        result = enc.fit_transform(self.hierarchical_cat_example, self.hierarchical_cat_example['target'])
        values = result['Compass'].values
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

    def test_hierarchical_smoothing_multi(self):

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=self.hierarchical_map, cols=['Compass', 'Speed', 'Animal'])
        result = enc.fit_transform(self.hierarchical_cat_example, self.hierarchical_cat_example['target'])

        values = result['Compass'].values
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

        values = result['Speed'].values
        self.assertAlmostEqual(0.6827, values[0], delta=1e-4)
        self.assertAlmostEqual(0.3962, values[4], delta=1e-4)
        self.assertAlmostEqual(0.4460, values[7], delta=1e-4)

        values = result['Animal'].values
        self.assertAlmostEqual(0.7887, values[0], delta=1e-4)
        self.assertAlmostEqual(0.3248, values[5], delta=1e-4)
        self.assertAlmostEqual(0.6190, values[11], delta=1e-4)
        self.assertAlmostEqual(0.1309, values[13], delta=1e-4)
        self.assertAlmostEqual(0.7381, values[15], delta=1e-4)

    def test_hierarchical_part_named_cols(self):

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=self.hierarchical_map, cols=['Compass'])
        result = enc.fit_transform(self.hierarchical_cat_example, self.hierarchical_cat_example['target'])

        values = result['Compass'].values
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)

        values = result['Speed'].values
        self.assertEqual('slow', values[0])

    def test_hierarchy_pandas_index(self):
        df = pd.DataFrame({
            'hello': ['a', 'b', 'c', 'a', 'a', 'b', 'c', 'd', 'd'],
            'world': [0, 1, 0, 0, 1, 0, 0, 1, 1]
        }, columns=pd.Index(['hello', 'world']))
        cols = df.select_dtypes(include='object').columns

        self.hierarchical_map = {
            'hello': {
                'A': ('a', 'b'),
                'B': ('c', 'd')
            },
        }

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=self.hierarchical_map, cols=cols)
        result = enc.fit_transform(df, df['world'])

        values = result['hello'].values
        self.assertAlmostEqual(0.3616, values[0], delta=1e-4)
        self.assertAlmostEqual(0.4541, values[1], delta=1e-4)
        self.assertAlmostEqual(0.2425, values[2], delta=1e-4)
        self.assertAlmostEqual(0.7425, values[7], delta=1e-4)

    def test_hierarchy_single_mapping(self):

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=self.hierarchical_map,
                                     cols=['Plant'])
        result = enc.fit_transform(self.hierarchical_cat_example, self.hierarchical_cat_example['target'])

        values = result['Plant'].values
        self.assertAlmostEqual(0.6828, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5, values[4], delta=1e-4)
        self.assertAlmostEqual(0.5, values[8], delta=1e-4)
        self.assertAlmostEqual(0.3172, values[12], delta=1e-4)

    def test_hierarchy_no_mapping(self):
        hierarchical_map = {
            'Plant': {
                'Rose': 'Rose',
                'Daisy': 'Daisy',
                'Daffodil': 'Daffodil',
                'Bluebell': 'Bluebell'
            }
        }

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map,
                                     cols=['Plant'])
        result = enc.fit_transform(self.hierarchical_cat_example, self.hierarchical_cat_example['target'])

        values = result['Plant'].values
        self.assertAlmostEqual(0.6828, values[0], delta=1e-4)
        self.assertAlmostEqual(0.5, values[4], delta=1e-4)
        self.assertAlmostEqual(0.5, values[8], delta=1e-4)
        self.assertAlmostEqual(0.3172, values[12], delta=1e-4)

    def test_hierarchy_error(self):
        hierarchical_map = {
            'Plant': {
                'Flower': {'Rose': ('Pink', 'Yellow', 'Red')},
                'Tree': 'Ash'
            }
        }
        with self.assertRaises(ValueError):
            encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map,
                                     cols=['Plant'])

    def test_hierarchy_multi_level(self):
        hierarchy_multi_level_df = pd.DataFrame(
            {
                'Animal': ['Cat', 'Cat', 'Dog', 'Dog', 'Dog', 'Osprey', 'Kite', 'Kite', 'Carp', 'Carp', 'Carp',
                            'Clownfish', 'Clownfish', 'Lizard', 'Snake', 'Snake'],
                'target': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
            }, columns=['Animal', 'target'])
        hierarchy_multi_level = {
            'Animal': {
                'Warm-Blooded':
                    {'Mammals': ('Cat', 'Dog'),
                     'Birds': ('Osprey', 'Kite'),
                     'Fish': ('Carp', 'Clownfish')
                    },
                'Cold-Blooded':
                    {'Reptiles': ('Lizard'),
                     'Amphibians': ('Snake', 'Frog')
                    }
            }}

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchy_multi_level,
                                     cols=['Animal'])
        result = enc.fit_transform(hierarchy_multi_level_df, hierarchy_multi_level_df['target'])

        values = result['Animal'].values
        self.assertAlmostEqual(0.6261, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9065, values[2], delta=1e-4)
        self.assertAlmostEqual(0.4107, values[5], delta=1e-4)
        self.assertAlmostEqual(0.3680, values[8], delta=1e-4)
        self.assertAlmostEqual(0.4626, values[11], delta=1e-4)
        self.assertAlmostEqual(0.2466, values[13], delta=1e-4)
        self.assertAlmostEqual(0.4741, values[14], delta=1e-4)


    def test_hierarchy_columnwise_compass(self):
        X, y = load_compass()
        cols = X.columns[~X.columns.str.startswith('HIER')]
        HIER_cols = X.columns[X.columns.str.startswith('HIER')]
        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols],
                                     cols=['compass'])
        result = enc.fit_transform(X[cols], y)

        values = result['compass'].values
        self.assertAlmostEqual(0.6226, values[0], delta=1e-4)
        self.assertAlmostEqual(0.9038, values[2], delta=1e-4)
        self.assertAlmostEqual(0.1766, values[5], delta=1e-4)
        self.assertAlmostEqual(0.4605, values[7], delta=1e-4)
        self.assertAlmostEqual(0.4033, values[11], delta=1e-4)


    def test_hierarchy_columnwise_postcodes(self):
        X, y = load_postcodes('binary')
        cols = X.columns[~X.columns.str.startswith('HIER')]
        HIER_cols = X.columns[X.columns.str.startswith('HIER')]
        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols],
                                     cols=['postcode'])
        result = enc.fit_transform(X[cols], y)

        values = result['postcode'].values
        self.assertAlmostEqual(0.7506, values[0], delta=1e-4)


    def test_hierarchy_columnwise_missing_level(self):
        X, y = load_postcodes('binary')
        HIER_cols = ['HIER_postcode_1', 'HIER_postcode_2', 'HIER_postcode_4']
        with self.assertRaises(ValueError):
            encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols],
                                   cols=['postcode'])


    def test_hierarchy_mapping_no_cols(self):
        hierarchical_map = {'Compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
        with self.assertRaises(ValueError):
            encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map)


    def test_hierarchy_mapping_cols_missing(self):
        X = ['N', 'N', 'NE', 'NE', 'NE', 'SE', 'SE', 'S', 'S', 'S', 'S', 'W', 'W', 'W', 'W', 'W']
        hierarchical_map = {'Compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
        y = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map,
                                                        cols=['Compass'])
        with self.assertRaises(ValueError):
            enc.fit_transform(X, y)
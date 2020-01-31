from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders as encoders
import pandas as pd
import numpy as np


class TestMEstimateEncoder(TestCase):

    def test_manual_targetencoder(self):
        x_k = ['A', 'A', 'A', 'B']
        y = [0, 0, 1, 0]
        x_t = ['A', 'A', 'B']

        # do LOO CV
        encoder = encoders.KFoldEncoder(K=len(x_k), min_samples_leaf=1, smoothing=1)
        encoder.fit(x_k, y)
        scored = encoder.transform(x_t)

        # get all folds
        res = []
        xx = pd.DataFrame({'x': ['A', 'A', 'B']})
        x = pd.DataFrame({'x': ['A', 'A', 'A'], 'y': [0, 0, 1]})
        res.append(encoders.TargetEncoder().fit(x.x, x.y).transform(xx.x).values)
        x = pd.DataFrame({'x': ['A', 'A', 'B'], 'y': [0, 1, 0]})
        res.append(encoders.TargetEncoder().fit(x.x, x.y).transform(xx.x).values)
        x = pd.DataFrame({'x': ['A', 'A', 'B'], 'y': [0, 1, 0]})
        res.append(encoders.TargetEncoder().fit(x.x, x.y).transform(xx.x).values)
        x = pd.DataFrame({'x': ['A', 'A', 'B'], 'y': [0, 0, 0]})
        res.append(encoders.TargetEncoder().fit(x.x, x.y).transform(xx.x).values)
        expected = np.mean(res, axis=0)

        self.assertEqual(scored.values.tolist(), expected.tolist())

import numpy as np
from unittest2 import TestCase
import category_encoders as encoders
import tests.helpers as th

# data definitions
X = th.create_dataset(n_rows=100)
np_y = np.random.randn(100) > 0.5

class TestGLMMEncoder(TestCase):
    def test_continuous(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical', 'categorical_int']
        enc = encoders.GLMMEncoder(cols=cols, binomial_target=False)
        enc.fit(X, np_y)
        th.verify_numeric(enc.transform(X))

    def test_binary(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical', 'categorical_int']
        enc = encoders.GLMMEncoder(cols=cols, binomial_target=True)
        enc.fit(X, np_y)
        th.verify_numeric(enc.transform(X))

from unittest import TestCase
import numpy as np
import category_encoders as encoders
import tests.helpers as th


class TestGLMMEncoder(TestCase):
    def test_continuous(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical', 'categorical_int']
        enc = encoders.GLMMEncoder(cols=cols, binomial_target=False)
        # TODO: fix this test IRL
        # enc.fit(X, np_y)
        #th.verify_numeric(enc.transform(X))

    def test_binary(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical', 'categorical_int']
        enc = encoders.GLMMEncoder(cols=cols, binomial_target=True)
        # TODO: fix this test IRL
        #enc.fit(X, np_y)
        #th.verify_numeric(enc.transform(X))

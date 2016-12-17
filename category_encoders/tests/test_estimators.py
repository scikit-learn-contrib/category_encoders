import unittest
from sklearn.utils.estimator_checks import check_transformer_general
from category_encoders import *

__author__ = 'willmcginnis'


class TestEncoders(unittest.TestCase):
    """
    """

    def test_general_transformers(self):
        check_transformer_general('hashing_encoder', HashingEncoder)
        check_transformer_general('backward_difference_encoder', BackwardDifferenceEncoder)
        check_transformer_general('binary_encoder', BinaryEncoder)
        check_transformer_general('helmert_encoder', HelmertEncoder)
        check_transformer_general('ordinal_encoder', OrdinalEncoder)
        check_transformer_general('polynomial_encoder', PolynomialEncoder)
        check_transformer_general('sum_coding', SumEncoder)
        check_transformer_general('one_hot', OneHotEncoder)
        check_transformer_general('basen', BaseNEncoder)
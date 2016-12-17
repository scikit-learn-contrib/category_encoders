"""

.. module:: category_encoders
  :synopsis:
  :platform:

"""

from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.basen import BaseNEncoder

__author__ = 'willmcginnis'

__all__ = [
    'BackwardDifferenceEncoder',
    'BinaryEncoder',
    'HashingEncoder',
    'HelmertEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'SumEncoder',
    'PolynomialEncoder',
    'BaseNEncoder'
]
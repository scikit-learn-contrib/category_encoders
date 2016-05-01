"""

.. module:: category_encoders
  :synopsis:
  :platform:

"""

from category_encoders.backward_difference import backward_difference_coding, BackwardDifferenceEncoder
from category_encoders.binary import binary, BinaryEncoder
from category_encoders.hashing import (
    hashing_trick,
    hashing_trick_4,
    hashing_trick_8,
    hashing_trick_16,
    hashing_trick_32,
    hashing_trick_64,
    hashing_trick_128,
    HashingEncoder
)
from category_encoders.helmert import helmert_coding, HelmertEncoder
from category_encoders.one_hot import one_hot
from sklearn.preprocessing import OneHotEncoder
from category_encoders.ordinal import ordinal_encoding, OrdinalEncoder
from category_encoders.sum_coding import sum_coding, SumEncoder
from category_encoders.polynomial import polynomial_coding, PolynomialEncoder

__author__ = 'willmcginnis'

__all__ = [
    'BackwardDifferenceEncoder',
    'BinaryEncoder',
    'HashingEncoder',
    'HelmertEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'SumEncoder',
    'PolynomialEncoder'
]
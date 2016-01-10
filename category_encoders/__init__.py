from category_encoders.backward_difference import backward_difference_coding
from category_encoders.binary import binary
from category_encoders.hashing import hashing_trick, hashing_trick_4, hashing_trick_8, hashing_trick_16, hashing_trick_32
from category_encoders.helmert import helmert_coding
from category_encoders.one_hot import one_hot
from category_encoders.ordinal import ordinal_encoding
from category_encoders.sum_coding import sum_coding
from category_encoders.polynomial import polynomial_coding

__author__ = 'willmcginnis'

__all__ = [
    'backward_difference_coding',
    'binary',
    'hashing_trick',
    'hashing_trick_4',
    'hashing_trick_8',
    'hashing_trick_16',
    'hashing_trick_32',
    'helmert_coding',
    'one_hot',
    'ordinal_encoding',
    'sum_coding',
    'polynomial_coding'
]
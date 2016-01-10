from encoders.backward_difference import backward_difference_coding
from encoders.binary import binary
from encoders.hashing import hashing_trick_2, hashing_trick_4, hashing_trick_8, hashing_trick_16, hashing_trick_32
from encoders.helmert import helmert_coding
from encoders.one_hot import one_hot
from encoders.ordinal import ordinal_encoding
from encoders.sum_coding import sum_coding
from encoders.polynomial import polynomial_coding

__author__ = 'willmcginnis'

__all__ = [
    'backward_difference_coding',
    'binary',
    'hashing_trick_2',
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
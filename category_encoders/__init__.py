"""Category encoders library.

.. module:: category_encoders
  :synopsis:
  :platform:

"""

from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.count import CountEncoder
from category_encoders.gray import GrayEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.quantile_encoder import QuantileEncoder, SummaryEncoder
from category_encoders.rankhot import RankHotEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder


def __getattr__(name):
    # GLMMEncoder pulls in statsmodels (which in turn pulls in patsy). Defer
    # the import so users who don't need GLMM aren't forced to install it.
    if name == 'GLMMEncoder':
        try:
            from category_encoders.glmm import GLMMEncoder
        except ImportError as e:
            raise ImportError(
                'GLMMEncoder requires statsmodels. Install with '
                "'pip install category_encoders[glmm]' or 'pip install statsmodels'."
            ) from e
        return GLMMEncoder
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__version__ = '2.8.1'

__author__ = 'willmcginnis', 'cmougan', 'paulwestenthanner'

__all__ = [
    'BackwardDifferenceEncoder',
    'BinaryEncoder',
    'GrayEncoder',
    'CountEncoder',
    'HashingEncoder',
    'HelmertEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'SumEncoder',
    'PolynomialEncoder',
    'BaseNEncoder',
    'LeaveOneOutEncoder',
    'TargetEncoder',
    'WOEEncoder',
    'MEstimateEncoder',
    'JamesSteinEncoder',
    'CatBoostEncoder',
    'GLMMEncoder',
    'QuantileEncoder',
    'SummaryEncoder',
    'RankHotEncoder',
]

import copy
import pandas as pd

__author__ = 'willmcginnis'


def hashing_trick_4(X_in):
    return hashing_trick_2(X_in, N=4)


def hashing_trick_8(X_in):
    return hashing_trick_2(X_in, N=8)


def hashing_trick_16(X_in):
    return hashing_trick_2(X_in, N=16)


def hashing_trick_32(X_in):
    return hashing_trick_2(X_in, N=32)


def hashing_trick_2(X_in, N=2):
    """
    A basic hashing implementation with configurable dimensionality/precision

    :param X_in:
    :return:
    """

    X = copy.deepcopy(X_in)

    def xform(x):
        tmp = [0 for _ in range(N)]
        tmp[hash(x) % N] = 1
        return pd.Series(tmp, index=cols)

    for col in X.columns.values:
        cols = [col + '_' + str(x) for x in range(N)]

        X[cols] = X[col].apply(xform)
        X = X.drop(col, axis=1)

    return X
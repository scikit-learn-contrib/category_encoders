import copy
import pandas as pd

__author__ = 'willmcginnis'


def hashing_trick_4(X_in):
    return hashing_trick(X_in, N=4)


def hashing_trick_8(X_in):
    return hashing_trick(X_in, N=8)


def hashing_trick_16(X_in):
    return hashing_trick(X_in, N=16)


def hashing_trick_32(X_in):
    return hashing_trick(X_in, N=32)


def hashing_trick_64(X_in):
    return hashing_trick(X_in, N=64)


def hashing_trick_128(X_in):
    return hashing_trick(X_in, N=128)


def hashing_trick(X_in, N=2):
    """
    A basic hashing implementation with configurable dimensionality/precision

    :param X_in:
    :return:
    """

    X = copy.deepcopy(X_in)

    def hash_fn(x):
        tmp = [0 for _ in range(N)]
        for val in x.values:
            tmp[hash(val) % N] += 1
        return pd.Series(tmp, index=cols)

    cols = ['col_%d' % d for d in range(N)]
    X = X.apply(hash_fn, axis=1)

    return X
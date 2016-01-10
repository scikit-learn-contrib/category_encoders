import copy
from patsy.highlevel import dmatrix

__author__ = 'willmcginnis'


def polynomial_coding(X_in):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        mod = dmatrix("C(%s, Poly)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X
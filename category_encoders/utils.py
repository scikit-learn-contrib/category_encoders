"""A collection of shared utilities for all encoders, not intended for external use."""

import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix

__author__ = 'willmcginnis'


def get_obj_cols(df):
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object':
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def convert_input(X):
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, list):
            X = pd.DataFrame(np.array(X))
        elif isinstance(X, (np.generic, np.ndarray)):
            X = pd.DataFrame(X)
        elif isinstance(X, csr_matrix):
            X = pd.DataFrame(X.todense())
        else:
            raise ValueError('Unexpected input type: %s' % (str(type(X))))

        X = X.convert_objects(convert_numeric=True)

    return X

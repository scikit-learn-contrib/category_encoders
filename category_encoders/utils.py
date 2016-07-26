"""A collection of shared utilities for all encoders, not intended for external use."""

import pandas as pd
import numpy as np

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
            X = X.convert_objects(convert_numeric=True)
        elif isinstance(X, (np.generic, np.ndarray)):
            X = pd.DataFrame(X)
            X = X.convert_objects(convert_numeric=True)
        else:
            raise ValueError('Unexpected input type: %s' % (str(type(X))))
    return X

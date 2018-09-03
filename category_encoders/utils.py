"""A collection of shared utilities for all encoders, not intended for external use."""

import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix

__author__ = 'willmcginnis'


def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object':
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def convert_input(X):
    """
    Unite data into a DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, list):
            X = pd.DataFrame(np.array(X))
        elif isinstance(X, (np.generic, np.ndarray)):
            X = pd.DataFrame(X)
        elif isinstance(X, csr_matrix):
            X = pd.DataFrame(X.todense())
        else:
            raise ValueError('Unexpected input type: %s' % (str(type(X))))

        X = X.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    return X


def get_generated_cols(X_original, X_transformed, to_transform):
    """
    Returns a list of the generated/transformed columns.

    Arguments:
        X_original: df
            the original (input) DataFrame.
        X_transformed: df
            the transformed (current) DataFrame.
        to_transform: [str]
            a list of columns that were transformed (as in the original DataFrame), commonly self.cols.

    Output:
        a list of columns that were transformed (as in the current DataFrame).
    """
    original_cols = set(X_original.columns)
    current_cols = set(X_transformed.columns)
    generated_cols = list(current_cols - (original_cols - set(to_transform)))

    return generated_cols
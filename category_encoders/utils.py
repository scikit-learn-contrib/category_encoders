"""A collection of shared utilities for all encoders, not intended for external use."""

import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix

__author__ = 'willmcginnis'


def convert_cols_to_list(cols):
    if isinstance(cols, pd.Series):
        return cols.tolist()
    elif isinstance(cols, np.ndarray):
        return cols.tolist()
    elif np.isscalar(cols):
        return [cols]
    elif isinstance(cols, set):
        return list(cols)
    elif isinstance(cols, tuple):
        return list(cols)
    elif pd.api.types.is_categorical(cols):
        return cols.astype(object).tolist()

    return cols


def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or is_category(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


def convert_input(X, columns=None, deep=False):
    """
    Unite data into a DataFrame.
    Objects that do not contain column names take the names from the argument.
    Optionally perform deep copy of the data.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, copy=deep)
        else:
            if columns is not None and np.size(X,1) != len(columns):
                raise ValueError('The count of the column names does not correspond to the count of the columns')
            if isinstance(X, list):
                X = pd.DataFrame(X, columns=columns, copy=deep)  # lists are always copied, but for consistency, we still pass the argument
            elif isinstance(X, (np.generic, np.ndarray)):
                X = pd.DataFrame(X, columns=columns, copy=deep)
            elif isinstance(X, csr_matrix):
                X = pd.DataFrame(X.todense(), columns=columns, copy=deep)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))
    elif deep:
        X = X.copy(deep=True)

    return X


def convert_input_vector(y, index):
    """
    Unite target data type into a Series.
    If the target is a Series or a DataFrame, we preserve its index.
    But if the target does not contain index attribute, we use the index from the argument.
    """
    if y is None:
        raise ValueError('Supervised encoders need a target for the fitting. The target cannot be None')
    if isinstance(y, pd.Series):
        return y
    elif isinstance(y, np.ndarray):
        if len(np.shape(y))==1:  # vector
            return pd.Series(y, name='target', index=index)
        elif len(np.shape(y))==2 and np.shape(y)[0]==1:  # single row in a matrix
            return pd.Series(y[0, :], name='target', index=index)
        elif len(np.shape(y))==2 and np.shape(y)[1]==1:  # single column in a matrix
            return pd.Series(y[:, 0], name='target', index=index)
        else:
            raise ValueError('Unexpected input shape: %s' % (str(np.shape(y))))
    elif np.isscalar(y):
        return pd.Series([y], name='target', index=index)
    elif isinstance(y, list):
        if len(y)==0 or (len(y)>0 and not isinstance(y[0], list)): # empty list or a vector
            return pd.Series(y, name='target', index=index, dtype=float)
        elif len(y)>0 and isinstance(y[0], list) and len(y[0])==1: # single row in a matrix
            flatten = lambda y: [item for sublist in y for item in sublist]
            return pd.Series(flatten(y), name='target', index=index)
        elif len(y)==1 and len(y[0])==0 and isinstance(y[0], list): # single empty column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=float)
        elif len(y)==1 and isinstance(y[0], list): # single column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=type(y[0][0]))
        else:
            raise ValueError('Unexpected input shape')
    elif isinstance(y, pd.DataFrame):
        if len(list(y))==0: # empty DataFrame
            return pd.Series(name='target', index=index, dtype=float)
        if len(list(y))==1: # a single column
            return y.iloc[:, 0]
        else:
            raise ValueError('Unexpected input shape: %s' % (str(y.shape)))
    else:
        return pd.Series(y, name='target', index=index)  # this covers tuples and other directly convertible types

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
    original_cols = list(X_original.columns)

    if len(to_transform) > 0:
        [original_cols.remove(c) for c in to_transform]

    current_cols = list(X_transformed.columns)
    if len(original_cols) > 0:
        [current_cols.remove(c) for c in original_cols]

    return current_cols

"""Helper functions that are used exclusively in the tests"""

import numpy as np
import random
import pandas as pd
import math


def verify_numeric(X_test):
    """
    Test that all attributes in the DataFrame are numeric.
    """
    _NUMERIC_KINDS = set('buifc')
    
    for dt in X_test.dtypes:
        assert(dt.kind in _NUMERIC_KINDS)


def create_array(n_rows=1000, extras=False, has_none=True):
    """
    Creates a numpy dataset with some categorical variables.
    """
    ds = [[
        random.random(),
        random.random(),
        random.choice(['A', 'B', 'C']),
        random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),
        random.choice(['A', 'B', 'C', None, np.nan]) if has_none else random.choice(['A', 'B', 'C']),
        random.choice(['A'])
    ] for _ in range(n_rows)]

    return np.array(ds)


def create_dataset(n_rows=1000, extras=False, has_missing=True, random_seed=2001):
    """
    Creates a dataset with some categorical variables.
    """
    random.seed(random_seed)
    ds = [[
        random.random(),                                                                        # Floats
        random.choice([float('nan'), float('inf'), float('-inf'), -0, 0, 1, -1, math.pi]),      # Floats with edge scenarios
        row,                                                                                    # Unique integers
        str(row),                                                                               # Unique strings
        random.choice(['A', 'B']) if extras else 'A',                                           # Invariant in the training data
        random.choice(['A', 'B_b', 'C_c_c']),                                                   # Strings with underscores to test reverse_dummies()
        random.choice(['A', 'B', 'C', np.NaN]) if has_missing else random.choice(['A', 'B', 'C']), # None
        random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),      # With a new string value
        random.choice([12, 43, -32]),                                                           # Number in the column name
        random.choice(['A', 'B', 'C']),                                                         # What is going to become the categorical column
        random.choice(['A', 'B', 'C', np.nan]),                                                 # Categorical with missing values
        random.choice([1, 2, 3])                                                                # Ordinal integers
    ] for row in range(n_rows)]

    df = pd.DataFrame(ds, columns=['float', 'float_edge', 'unique_int', 'unique_str', 'invariant', 'underscore', 'none', 'extra', 321, 'categorical', 'na_categorical', 'categorical_int'])
    df['categorical'] = pd.Categorical(df['categorical'], categories=['A', 'B', 'C'])
    df['na_categorical'] = pd.Categorical(df['na_categorical'], categories=['A', 'B', 'C'])
    df['categorical_int'] = pd.Categorical(df['categorical_int'], categories=[1, 2, 3])
    return df


def verify_inverse_transform(x, x_inv):
    """
    Verify x is equal to x_inv. The test returns true for NaN.equals(NaN) as it should.
    """
    assert x.equals(x_inv)


def deep_round(A, ndigits=5):
    """
    Rounds numbers in a list of lists. Useful for approximate equality testing.
    """
    return [[round(val, ndigits) for val in sublst] for sublst in A]

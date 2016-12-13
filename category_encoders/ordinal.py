"""Ordinal or label encoding"""

import pandas as pd
import copy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import random
from category_encoders.utils import get_obj_cols, convert_input

__author__ = 'willmcginnis'


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features as ordinal, in one ordered feature

    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array)
    mapping: dict
        a mapping of class to label to use for the encoding, optional.
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1

    Example
    -------
    >>>from category_encoders import *
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>enc = OrdinalEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>>numeric_dataset = enc.transform(X)
    >>>print(numeric_dataset.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null int64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null int64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(2)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    http://www.ats.ucla.edu/stat/r/library/contrast_coding.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """
    def __init__(self, verbose=0, mapping=None, cols=None, drop_invariant=False, return_df=True, impute_missing=True, handle_unknown='impute'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        _, categories = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        self.mapping = categories

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Will use the mapping (if available) and the column list (if available, otherwise every column) to encode the
        data ordinally.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        if not self.cols:
            return X

        X, _ = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def ordinal_encoding(X_in, mapping=None, cols=None, impute_missing=True, handle_unknown='impute'):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.
        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        mapping_out = []
        if mapping is not None:
            for switch in mapping:
                X[str(switch.get('col')) + '_tmp'] = np.nan
                for category in switch.get('mapping'):
                    X.loc[X[switch.get('col')] == category[0], str(switch.get('col')) + '_tmp'] = str(category[1])
                del X[switch.get('col')]
                X.rename(columns={str(switch.get('col')) + '_tmp': switch.get('col')}, inplace=True)

                if impute_missing:
                    if handle_unknown == 'impute':
                        X[switch.get('col')].fillna(-1, inplace=True)
                    elif handle_unknown == 'error':
                        if X[~X['D'].isin([str(x[1]) for x in switch.get('mapping')])].shape[0] > 0:
                            raise ValueError('Unexpected categories found in %s' % (switch.get('col'), ))

                try:
                    X[switch.get('col')] = X[switch.get('col')].astype(int).reshape(-1, )
                except ValueError as e:
                    X[switch.get('col')] = X[switch.get('col')].astype(float).reshape(-1, )
        else:
            for col in cols:
                categories = list(set(X[col].values))
                random.shuffle(categories)

                X[str(col) + '_tmp'] = np.nan
                for idx, val in enumerate(categories):
                    X.loc[X[col] == val, str(col) + '_tmp'] = str(idx)
                del X[col]
                X.rename(columns={str(col) + '_tmp': col}, inplace=True)

                if impute_missing:
                    if handle_unknown == 'impute':
                        X[col].fillna(-1, inplace=True)

                try:
                    X[col] = X[col].astype(int).reshape(-1, )
                except ValueError as e:
                    X[col] = X[col].astype(float).reshape(-1, )

                mapping_out.append({'col': col, 'mapping': [(x[1], x[0]) for x in list(enumerate(categories))]},)

        return X, mapping_out
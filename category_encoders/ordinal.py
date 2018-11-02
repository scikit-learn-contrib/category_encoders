"""Ordinal or label encoding"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util

__author__ = 'willmcginnis'


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features as ordinal, in one ordered feature.

    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
   mapping: list of dict
        a mapping of class to label to use for the encoding, optional.
        the dict contains the keys 'col' and 'mapping'.
        the value of 'col' should be the feature name.
        the value of 'mapping' should be a list of tuples of format (original_label, encoded_label).
        example mapping: [{'col': 'col1', 'mapping': [(None, 0), ('a', 1), ('b', 2)]}]
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which will impute the category -1.
    handle_missing: str
        options are 'error', 'return_nan', and 'value, default to 'value', which treat nan as a category at fit time,
        or -2 at transform time if nan is not a category during fit.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OrdinalEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
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
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """

    def __init__(self, verbose=0, mapping=None, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._dim = None

    @property
    def category_mapping(self):
        return self.mapping

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
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

        _, categories = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing
        )
        self.mapping = categories

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Will use the mapping (if available) and the column list (if available, otherwise every column) to encode the
        data ordinarily.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        X, _ = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        return X if self.return_df else X.values

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """
        X = X_in.copy(deep=True)

        # first check the type
        X = util.convert_input(X)

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "set as False when transform data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        if self.handle_unknown == 'value':
            for col in self.cols:
                if any(X[col] == -1):
                    raise ValueError("inverse_transform is not supported because transform impute "
                                     "the unknown category -1 when encode %s" % (col,))

        if self.handle_unknown == 'return_nan':
            for col in self.cols:
                if X[col].isnull().any():
                    raise ValueError("inverse_transform is not supported because transform impute "
                                     "the unknown category nan when encode %s" % (col,))

        for switch in self.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.get_values())
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X if self.return_df else X.values

    @staticmethod
    def ordinal_encoding(X_in, mapping=None, cols=None, handle_unknown='value', handle_missing='valie'):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.
        """

        return_nan_series = pd.Series(data=[np.nan], index=[-2])

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                column = switch.get('col')
                X[column] = X[column].map(switch['mapping'])

                try:
                    X[column] = X[column].astype(int)
                except ValueError as e:
                    X[column] = X[column].astype(float)

                if handle_unknown == 'value':
                    X[column].fillna(0, inplace=True)
                elif handle_unknown == 'error':
                    missing = X[column].isnull()
                    if any(missing):
                        raise ValueError('Unexpected categories found in column %s' % column)

                if handle_missing == 'return_nan':
                    X[column] = X[column].map(return_nan_series).where(X[column] == -2, X[column])

        else:
            mapping_out = []
            for col in cols:

                if util.is_category(X[col].dtype):
                    categories = X[col].cat.categories
                else:
                    categories = [x if x is not None else np.nan for x in pd.unique(X[col].values)]

                data = {}

                if handle_missing == 'value':
                    data[np.nan] = -2

                for i in range(len(categories)):
                    data[categories[i]] = i + 1

                if handle_missing == 'return_nan':
                    data[np.nan] = -2

                mapping = pd.Series(data)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out

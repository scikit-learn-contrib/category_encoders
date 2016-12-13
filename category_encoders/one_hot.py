"""One-hot or dummy coding"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import get_obj_cols, convert_input

__author__ = 'willmcginnis'


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Onehot (or dummy) coding for categorical features, produces one feature per category, each binary.

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
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1. Warning: if
        impute is used, an extra column will be added in if the transform matrix has unknown categories.  This can causes
        unexpected changes in dimension in some cases.

    Example
    -------
    >>>from category_encoders import *
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>enc = OneHotEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>>numeric_dataset = enc.transform(X)
    >>>print(numeric_dataset.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 22 columns):
    CHAS_0     506 non-null int64
    CHAS_1     506 non-null int64
    RAD_0      506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
    RAD_5      506 non-null int64
    RAD_6      506 non-null int64
    RAD_7      506 non-null int64
    RAD_8      506 non-null int64
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(11)
    memory usage: 87.0 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    http://www.ats.ucla.edu/stat/r/library/contrast_coding.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """
    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True, handle_unknown='impute'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown

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

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

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

        X = self.ordinal_encoder.transform(X)

        X = self.get_dummies(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def get_dummies(X_in, cols=None):
        """
        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values
            pass_thru = []
        else:
            pass_thru = [col for col in X.columns.values if col not in cols]

        bin_cols = []
        for col in cols:
            classes = [x for x in set(X[col].values.tolist()) if np.isfinite(x)]
            for class_ in classes:
                n_col_name = str(col) + '_%s' % (class_, )
                X[n_col_name] = X[col] == class_
                bin_cols.append(n_col_name)

        X = X.reindex(columns=bin_cols + pass_thru)

        # convert all of the bools into integers.
        for col in bin_cols:
            X[col] = X[col].astype(int)

        return X
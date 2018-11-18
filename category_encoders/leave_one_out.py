"""Leave one out coding"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'hbghhy'


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """Leave one out coding for categorical features.

    This is very similar to target encoding, but excludes the current row's
    target when calculating the mean target for a level to reduce the effect
    of outliers.

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
    handle_unknown: str
        options are 'error', 'ignore' and 'value', defaults to 'value', which will impute the target mean.
    sigma: float
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
        sigma gives the standard deviation (spread or "width") of the normal distribution.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = LeaveOneOutEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] Strategies to encode categorical variables with many categories. from
    https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154.
    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, sigma=None):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.use_default_cols = cols is None # if True, even a repeated call of fit() will select string columns from X
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.random_state = random_state
        self.sigma = sigma

    def fit(self, X, y, **kwargs):
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
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0].astype(float)
        else:
            y = pd.Series(y, name='target', dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.use_default_cols:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

        categories = self.fit_leave_one_out(
            X, y,
            cols=self.cols
        )
        self.mapping = categories

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X, y=None):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)

            

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

        # if we are encoding the training data, we have to check the target
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0].astype(float)
            else:
                y = pd.Series(y, name='target', dtype=float)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not self.cols:
            return X
        X = self.transform_leave_one_out(
            X, y,
            mapping=self.mapping,
            handle_unknown=self.handle_unknown
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
             transform(X, y)
        and not with:
            transform(X)
        """
        return self.fit(X, y, **fit_params).transform(X, y)

    def fit_leave_one_out(self, X_in, y, cols=None):
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        self._mean = y.mean()
        return {col: y.groupby(X[col]).agg(['sum', 'count']) for col in cols}

    def transform_leave_one_out(self, X_in, y, mapping=None, handle_unknown='value'):
        """
        Leave one out encoding uses a single column of floats to represent the means of the target variables.
        """

        X = X_in.copy(deep=True)
        random_state_ = check_random_state(self.random_state)

        for col, colmap in mapping.items():
            level_notunique = colmap['count'] > 1
            if y is None:    # Replace level with its mean target; if level occurs only once, use global mean
                level_means = (colmap['sum'] / colmap['count']).where(level_notunique, self._mean)
                X[col] = X[col].map(level_means)
            else:            # Replace level with its mean target, calculated excluding this row's target
                # The y (target) mean for this level is normally just the sum/count;
                # excluding this row's y, it's (sum - y) / (count - 1)
                level_means = (X[col].map(colmap['sum']) - y) / (X[col].map(colmap['count']) - 1)
                # The 'where' fills in singleton levels (count = 1 -> div by 0) with the global mean
                X[col] = level_means.where(X[col].map(colmap['count'][level_notunique]).notnull(), self._mean)

            if handle_unknown == 'value':
                X[col].fillna(self._mean, inplace=True)
            elif handle_unknown == 'error':
                if X[col].isnull().any():
                    raise ValueError('Unexpected categories found in column %s' % col)

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1., self.sigma, X[col].shape[0])

        return X

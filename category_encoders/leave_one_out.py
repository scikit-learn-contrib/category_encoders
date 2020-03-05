"""Leave one out coding"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'hbghhy'


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """Leave one out coding for categorical features.

    This is very similar to target encoding but excludes the current row's
    target when calculating the mean target for a level to reduce the effect
    of outliers.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
    sigma: float
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing
        data are untouched). Sigma gives the standard deviation (spread or "width") of the normal distribution.
        The optimal value is commonly between 0.05 and 0.6. The default is to not add noise, but that leads
        to significantly suboptimal results.


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

    .. [1] Strategies to encode categorical variables with many categories, from
    https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154.

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, sigma=None):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.use_default_cols = cols is None  # if True, even a repeated call of fit() will select string columns from X
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._mean = None
        self.random_state = random_state
        self.sigma = sigma
        self.feature_names = None

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

        # unite the input into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index).astype(float)

        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.use_default_cols:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        categories = self.fit_leave_one_out(
            X, y,
            cols=self.cols
        )
        self.mapping = categories

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                          "Not found in generated cols.\n{}".format(e))

        return self

    def transform(self, X, y=None, override_return_df=False):
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
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # if we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index).astype(float)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X
        X = self.transform_leave_one_out(
            X, y,
            mapping=self.mapping
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
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

        # the interface requires 'y=None' in the signature but we need 'y'
        if y is None:
            raise(TypeError, 'fit_transform() missing argument: ''y''')

        return self.fit(X, y, **fit_params).transform(X, y)

    def fit_leave_one_out(self, X_in, y, cols=None):
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        self._mean = y.mean()

        return {col: self.fit_column_map(X[col], y) for col in cols}

    def fit_column_map(self, series, y):
        category = pd.Categorical(series)

        categories = category.categories
        codes = category.codes.copy()

        codes[codes == -1] = len(categories)
        categories = np.append(categories, np.nan)

        return_map = pd.Series(dict([(code, category) for code, category in enumerate(categories)]))

        result = y.groupby(codes).agg(['sum', 'count'])
        return result.rename(return_map)

    def transform_leave_one_out(self, X_in, y, mapping=None):
        """
        Leave one out encoding uses a single column of floats to represent the means of the target variables.
        """

        X = X_in.copy(deep=True)
        random_state_ = check_random_state(self.random_state)

        for col, colmap in mapping.items():
            level_notunique = colmap['count'] > 1

            unique_train = colmap.index
            unseen_values = pd.Series([x for x in X[col].unique() if x not in unique_train], dtype=unique_train.dtype)

            is_nan = X[col].isnull()
            is_unknown_value = X[col].isin(unseen_values.dropna().astype(object))

            if X[col].dtype.name == 'category': # Pandas 0.24 tries hard to preserve categorical data type
                X[col] = X[col].astype(str)

            if self.handle_unknown == 'error' and is_unknown_value.any():
                raise ValueError('Columns to be encoded can not contain new values')

            if y is None:    # Replace level with its mean target; if level occurs only once, use global mean
                level_means = (colmap['sum'] / colmap['count']).where(level_notunique, self._mean)
                X[col] = X[col].map(level_means)
            else:            # Replace level with its mean target, calculated excluding this row's target
                # The y (target) mean for this level is normally just the sum/count;
                # excluding this row's y, it's (sum - y) / (count - 1)
                level_means = (X[col].map(colmap['sum']) - y) / (X[col].map(colmap['count']) - 1)
                # The 'where' fills in singleton levels (count = 1 -> div by 0) with the global mean
                X[col] = level_means.where(X[col].map(colmap['count'][level_notunique]).notnull(), self._mean)

            if self.handle_unknown == 'value':
                X.loc[is_unknown_value, col] = self._mean
            elif self.handle_unknown == 'return_nan':
                X.loc[is_unknown_value, col] = np.nan

            if self.handle_missing == 'value':
                X.loc[is_nan & unseen_values.isnull().any(), col] = self._mean
            elif self.handle_missing == 'return_nan':
                X.loc[is_nan, col] = np.nan

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1., self.sigma, X[col].shape[0])

        return X

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """

        if not isinstance(self.feature_names, list):
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names

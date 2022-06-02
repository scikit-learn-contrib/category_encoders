"""CatBoost coding"""

import numpy as np
import pandas as pd
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class CatBoostEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """CatBoost Encoding for categorical features.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    This is very similar to leave-one-out encoding, but calculates the
    values "on-the-fly". Consequently, the values naturally vary
    during the training phase and it is not necessary to add random noise.

    Beware, the training data have to be randomly permutated. E.g.:

        # Random permutation
        perm = np.random.permutation(len(X))
        X = X.iloc[perm].reset_index(drop=True)
        y = y.iloc[perm].reset_index(drop=True)

    This is necessary because some data sets are sorted based on the target
    value and this coder encodes the features on-the-fly in a single pass.

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
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
        sigma gives the standard deviation (spread or "width") of the normal distribution.
    a: float
        additive smoothing (it is the same variable as "m" in m-probability estimate). By default set to 1.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = CatBoostEncoder(cols=['CHAS', 'RAD']).fit(X, y)
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

    .. [1] Transforming categorical features to numerical features, from
    https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/

    .. [2] CatBoost: unbiased boosting with categorical features, from
    https://arxiv.org/abs/1706.09516

    """
    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, sigma=None, a=1):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.mapping = None
        self._mean = None
        self.random_state = random_state
        self.sigma = sigma
        self.a = a

    def _fit(self, X, y, **kwargs):
        X = X.copy(deep=True)

        self._mean = y.mean()
        self.mapping = {col: self._fit_column_map(X[col], y) for col in self.cols}

    def _transform(self, X, y=None):
        random_state_ = check_random_state(self.random_state)

        # Prepare the data
        if y is not None:
            # Convert bools to numbers (the target must be summable)
            y = y.astype('double')

        for col, colmap in self.mapping.items():
            level_notunique = colmap['count'] > 1

            unique_train = colmap.index
            unseen_values = pd.Series([x for x in X[col].unique() if x not in unique_train], dtype=unique_train.dtype)

            is_nan = X[col].isnull()
            is_unknown_value = X[col].isin(unseen_values.dropna().astype(object))

            if self.handle_unknown == 'error' and is_unknown_value.any():
                raise ValueError('Columns to be encoded can not contain new values')

            if y is None:    # Replace level with its mean target; if level occurs only once, use global mean
                level_means = ((colmap['sum'] + self._mean * self.a) / (colmap['count'] + self.a)).where(level_notunique, self._mean)
                X[col] = X[col].map(level_means)
            else:
                # Simulation of CatBoost implementation, which calculates leave-one-out on the fly.
                # The nice thing about this is that it helps to prevent overfitting. The bad thing
                # is that CatBoost uses many iterations over the data. But we run just one iteration.
                # Still, it works better than leave-one-out without any noise.
                # See:
                #   https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
                # Cumsum does not work nicely with None (while cumcount does).
                # As a workaround, we cast the grouping column as string.
                # See: issue #209
                temp = y.groupby(X[col].astype(str)).agg(['cumsum', 'cumcount'])
                X[col] = (temp['cumsum'] - y + self._mean * self.a) / (temp['cumcount'] + self.a)

            if self.handle_unknown == 'value':
                if X[col].dtype.name == 'category':
                    X[col] = X[col].astype(float)
                X.loc[is_unknown_value, col] = self._mean
            elif self.handle_unknown == 'return_nan':
                X.loc[is_unknown_value, col] = np.nan

            if self.handle_missing == 'value':
                # only set value if there are actually missing values.
                # In case of pd.Categorical columns setting values that are not seen in pd.Categorical gives an error.
                nan_cond = is_nan & unseen_values.isnull().any()
                if nan_cond.any():
                    X.loc[nan_cond, col] = self._mean
            elif self.handle_missing == 'return_nan':
                X.loc[is_nan, col] = np.nan

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1., self.sigma, X[col].shape[0])

        return X

    def _more_tags(self):
        tags = super()._more_tags()
        tags["predict_depends_on_y"] = True
        return tags

    def _fit_column_map(self, series, y):
        category = pd.Categorical(series)

        categories = category.categories
        codes = category.codes.copy()

        codes[codes == -1] = len(categories)
        categories = np.append(categories, np.nan)

        return_map = pd.Series(dict(enumerate(categories)))

        result = y.groupby(codes).agg(['sum', 'count'])
        return result.rename(return_map)

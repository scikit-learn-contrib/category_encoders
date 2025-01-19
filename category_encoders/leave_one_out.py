"""Leave one out coding."""

import numpy as np
import pandas as pd
from sklearn.utils.random import check_random_state

import category_encoders.utils as util

__author__ = 'hbghhy'


class LeaveOneOutEncoder( util.SupervisedTransformerMixin,util.BaseEncoder):
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
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value',
         which returns the target mean.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value',
         which returns the target mean.
    sigma: float
        adds normal (Gaussian) distribution noise into training data in order to decrease
        overfitting (testing data are untouched). Sigma gives the standard deviation
        (spread or "width") of the normal distribution. The optimal value is commonly
        between 0.05 and 0.6.
        The default is to not add noise, but that leads to significantly suboptimal results.


    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = LeaveOneOutEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   Id           1460 non-null   float64
     1   MSSubClass   1460 non-null   float64
     2   MSZoning     1460 non-null   object
     3   LotFrontage  1201 non-null   float64
     4   YearBuilt    1460 non-null   float64
     5   Heating      1460 non-null   float64
     6   CentralAir   1460 non-null   float64
    dtypes: float64(6), object(1)
    memory usage: 80.0+ KB
    None

    References
    ----------

    .. [1] Originally by Owen Zhang (reference broken), another short explanation at:
    https://datascience.stackexchange.com/questions/10839/what-is-difference-between-one-hot-encoding-and-leave-one-out-encoding

    """

    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_unknown='value',
        handle_missing='value',
        random_state=None,
        sigma=None,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.mapping = None
        self._mean = None
        self.random_state = random_state
        self.sigma = sigma

    def _fit(self, X, y, **kwargs):
        y = y.astype(float)
        categories = self.fit_leave_one_out(X, y, cols=self.cols)
        self.mapping = categories

    def _transform(self, X, y=None):
        if y is not None:
            y = y.astype(float)

        X = self.transform_leave_one_out(X, y, mapping=self.mapping)
        return X

    def __sklearn_tags__(self) -> util.EncoderTags:
        """Set scikit transformer tags."""
        tags = super().__sklearn_tags__()
        tags.predict_depends_on_y = True
        return tags

    def fit_leave_one_out(
        self, X_in: pd.DataFrame, y: pd.Series, cols=None
    ) -> dict[str, pd.Series]:
        """Fit leave one out encoding."""
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns

        self._mean = y.mean()

        return {col: self._fit_column_map(X[col], y) for col in cols}

    @staticmethod
    def _fit_column_map(series: pd.Series, y: pd.Series) -> pd.Series:
        category = pd.Categorical(series)

        categories = category.categories
        codes = category.codes.copy()

        codes[codes == -1] = len(categories)
        categories = np.append(categories, np.nan)

        return_map = pd.Series(dict(enumerate(categories)))

        result = y.groupby(codes).agg(['sum', 'count'])
        return result.rename(return_map)

    def transform_leave_one_out(self, X: pd.DataFrame, y: pd.Series | None, mapping=None):
        """Apply leave-one-out-encoding to a dataframe.

        If a target is given the lable-mean is calculated without the target (left out).
        Otherwise, the label mean from the fit step is taken.
        """
        random_state_ = check_random_state(self.random_state)

        for col, colmap in mapping.items():
            level_notunique = colmap['count'] > 1

            unique_train = colmap.index
            unseen_values = pd.Series(
                [x for x in X[col].unique() if x not in unique_train], dtype=unique_train.dtype
            )

            is_nan = X[col].isna()
            is_unknown_value = X[col].isin(unseen_values.dropna().astype(object))

            if (
                X[col].dtype.name == 'category'
            ):  # Pandas 0.24 tries hard to preserve categorical data type
                index_dtype = X[col].dtype.categories.dtype
                X[col] = X[col].astype(index_dtype)

            if self.handle_unknown == 'error' and is_unknown_value.any():
                raise ValueError('Columns to be encoded can not contain new values')

            if (
                y is None
            ):  # Replace level with its mean target; if level occurs only once, use global mean
                level_means = (colmap['sum'] / colmap['count']).where(level_notunique, self._mean)
                X[col] = X[col].map(level_means)
            else:  # Replace level with its mean target, calculated excluding this row's target
                # The y (target) mean for this level is normally just the sum/count;
                # excluding this row's y, it's (sum - y) / (count - 1)
                level_means = (X[col].map(colmap['sum']) - y) / (X[col].map(colmap['count']) - 1)
                # The 'where' fills in singleton levels (count = 1 -> div by 0) with the global mean
                X[col] = level_means.where(
                    X[col].map(colmap['count'][level_notunique]).notna(), self._mean
                )

            if self.handle_unknown == 'value':
                X.loc[is_unknown_value, col] = self._mean
            elif self.handle_unknown == 'return_nan':
                X.loc[is_unknown_value, col] = np.nan

            if self.handle_missing == 'value':
                X.loc[is_nan & unseen_values.isna().any(), col] = self._mean
            elif self.handle_missing == 'return_nan':
                X.loc[is_nan, col] = np.nan

            if self.sigma is not None and y is not None:
                X[col] = X[col] * random_state_.normal(1.0, self.sigma, X[col].shape[0])

        return X

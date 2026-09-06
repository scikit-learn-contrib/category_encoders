"""Weight of Evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.random import check_random_state

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'Jan Motl'


class WOEEncoder( util.SupervisedTransformerMixin,util.BaseEncoder):
    """Weight of Evidence coding for categorical features.

    Supported targets: binomial. For polynomial target support, see PolynomialWrapper.

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
    handle_missing: str, int, float or callable
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
        A number is used as the encoded value for missing values that were not seen at fit time,
        and a callable fn(value, mapping) is evaluated once per column when the mapping is
        finalized during fit.
    handle_unknown: str, int, float or callable
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
        A number is used as the encoded value for unseen categories, and a callable
        fn(value, mapping) is evaluated once per column when the mapping is finalized during fit.
    randomized: bool,
        adds Gaussian regularization noise to the encoded values during fit
        to decrease overfitting. The noise is multiplicative — encoded values
        are scaled by ``N(1, sigma)`` — so it is centered on 1, not 0
        (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    regularization: float
        the purpose of regularization is mostly to prevent division by zero.
        When regularization is 0, you may encounter division by zero.

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
    >>> y = bunch.target > 200000
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = WOEEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
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

    .. [1] Weight of Evidence (WOE) and Information Value Explained, from
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    """

    prefit_ordinal = True
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
        randomized=False,
        sigma=0.05,
        regularization=1.0,
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            min_group_size=min_group_size,
            min_group_name=min_group_name,
            combine_min_nan_groups=combine_min_nan_groups,
        )
        self.ordinal_encoder = None
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization

    def _fit(self, X, y, **kwargs):
        # The label must be binary with values {0,1}
        y = pd.Series(y)
        unique = y.unique()
        if len(unique) != 2:
            raise ValueError(
                'The target column y must be binary. But the target contains '
                + str(len(unique))
                + ' unique value(s).'
            )
        if y.isna().any():
            raise ValueError('The target column y must not contain missing values.')
        if np.max(unique) < 1:
            msg = (
                'The target column y must be binary with values {0, 1}. '
                'Value 1 was not found in the target.'
            )
            raise ValueError(msg)
        if np.min(unique) > 0:
            msg = (
                'The target column y must be binary with values {0, 1}. '
                'Value 0 was not found in the target.'
            )
            raise ValueError(msg)

        # One ordinal pass: derive the category mapping read-only, then fit the
        # helper encoder from that mapping. With a mapping supplied, fit only
        # binds column metadata (the #503 contract: the mapping, not the data,
        # is the source of truth), so it needs no full-frame pass over the data.
        _, categories = OrdinalEncoder.ordinal_encoding(
            X, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            mapping=categories,
            handle_unknown='value',
            handle_missing='value',
        )
        self.ordinal_encoder.fit(X.iloc[0:0])

        # Ordinal-encode one disposable copy in place for training.
        X_ordinal = X.copy(deep=True)
        OrdinalEncoder.ordinal_encoding(
            X_ordinal,
            mapping=self.ordinal_encoder.mapping,
            cols=self.ordinal_encoder.cols,
            handle_unknown=self.ordinal_encoder.handle_unknown,
            handle_missing=self.ordinal_encoder.handle_missing,
            index_start=self.ordinal_encoder.index_start,
        )

        # Training
        self.mapping = self._train(X_ordinal, y)

    def _transform(self, X, y=None):
        # X is the private copy made by the transformer API (the #503 contract):
        # ordinal-encode it in place instead of stacking another wrapper copy.
        OrdinalEncoder.ordinal_encoding(
            X,
            mapping=self.ordinal_encoder.mapping,
            cols=self.ordinal_encoder.cols,
            handle_unknown=self.ordinal_encoder.handle_unknown,
            handle_missing=self.ordinal_encoder.handle_missing,
            index_start=self.ordinal_encoder.index_start,
        )

        if self.handle_unknown == 'error':
            if X[self.cols].isin([util.UNKNOWN_SENTINEL]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)
        return X

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['sum', 'count'])  # Count of x_{i,+} and x_i

            # Create a new column with regularized WOE.
            # Regularization helps to avoid division by zero.
            # Pre-calculate WOEs because logarithms are slow.
            nominator = (stats['sum'] + self.regularization) / (self._sum + 2 * self.regularization)
            denominator = ((stats['count'] - stats['sum']) + self.regularization) / (
                self._count - self._sum + 2 * self.regularization
            )
            woe = np.log(nominator / denominator)

            # Ignore unique values. This helps to prevent overfitting on id-like columns.
            woe[stats['count'] == 1] = 0

            woe = util.finalize_encoding_mapping(
                woe, values, self.handle_unknown, self.handle_missing, 0
            )

            # Store WOE for transform() function
            mapping[col] = woe

        return mapping

    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = X[col] * random_state_generator.normal(1.0, self.sigma, X[col].shape[0])

        return X

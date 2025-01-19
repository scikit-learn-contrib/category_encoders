"""Target Encoder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'chappers'


class TargetEncoder( util.SupervisedTransformerMixin,util.BaseEncoder):
    """Target encoding for categorical features.

    Supported targets: binomial and continuous.
    For polynomial target support, see PolynomialWrapper.

    For the case of categorical target: features are replaced with a blend of posterior
    probability of the target given particular categorical value and the prior probability
    of the target over all the training data.

    For the case of continuous target: features are replaced with a blend of the expected value
    of the target given particular categorical value and the expected value of the
    target over all the training data.

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
    min_samples_leaf: int
        For regularization the weighted average between category mean and global mean is taken.
        The weight is an S-shaped curve between 0 and 1 with the number of samples for a category
        on the x-axis. The curve reaches 0.5 at min_samples_leaf.
        (parameter k in the original paper)
    smoothing: float
        smoothing effect to balance categorical average vs prior. Higher value means stronger
        regularization. The value must be strictly bigger than 0.
        Higher values mean a flatter S-curve (see min_samples_leaf).
    hierarchy: dict or dataframe
        A dictionary or a dataframe to define the hierarchy for mapping.

        If a dictionary, this contains a dict of columns to map into hierarchies.
        Dictionary key(s) should be the column name from X which requires mapping.
        For multiple hierarchical maps, this should be a dictionary of dictionaries.

        If dataframe: a dataframe defining columns to be used for the hierarchies.
        Column names must take the form:
            HIER_colA_1, ... HIER_colA_N, HIER_colB_1, ... HIER_colB_M, ...
        where [colA, colB, ...] are given columns in cols list.
        1:N and 1:M define the hierarchy for each column where 1 is the highest hierarchy
        (top of the tree).  A single column or multiple can be used, as relevant.

    Examples
    --------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> y = bunch.target > 200000
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = TargetEncoder(cols=['CentralAir', 'Heating'], min_samples_leaf=20, smoothing=10).fit(
    ...     X, y
    ... )
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

    >>> from category_encoders.datasets import load_compass
    >>> X, y = load_compass()
    >>> hierarchical_map = {'compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
    >>> enc = TargetEncoder(
    ...     verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['compass']
    ... ).fit(X.loc[:, ['compass']], y)
    >>> hierarchy_dataset = enc.transform(X.loc[:, ['compass']])
    >>> print(hierarchy_dataset['compass'].values)
    [0.62263617 0.62263617 0.90382995 0.90382995 0.90382995 0.17660024
     0.17660024 0.46051953 0.46051953 0.46051953 0.46051953 0.40332791
     0.40332791 0.40332791 0.40332791 0.40332791]
    >>> X, y = load_postcodes('binary')
    >>> cols = ['postcode']
    >>> HIER_cols = ['HIER_postcode_1', 'HIER_postcode_2', 'HIER_postcode_3', 'HIER_postcode_4']
    >>> enc = TargetEncoder(
    ...     verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols], cols=['postcode']
    ... ).fit(X['postcode'], y)
    >>> hierarchy_dataset = enc.transform(X['postcode'])
    >>> print(hierarchy_dataset.loc[0:10, 'postcode'].values)
    [0.75063473 0.90208756 0.88328833 0.77041254 0.68891504 0.85012847
    0.76772574 0.88742357 0.7933824  0.63776756 0.9019973 ]

    References
    ----------

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification
    and Prediction Problems, from https://dl.acm.org/citation.cfm?id=507538

    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_missing: str = 'value',
        handle_unknown: str = 'value',
        min_samples_leaf: int = 20,
        smoothing: float = 10,
        hierarchy: dict = None,
    ) -> None:
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.mapping = None
        self._mean = None
        # @ToDo create a function to check the hierarchy
        if isinstance(hierarchy, (dict, pd.DataFrame)) and cols is None:
            raise ValueError('Hierarchy is defined but no columns are named for encoding')
        if isinstance(hierarchy, dict):
            self.hierarchy = {}
            self.hierarchy_depth = {}
            for switch in hierarchy:
                flattened_hierarchy = util.flatten_reverse_dict(hierarchy[switch])
                hierarchy_check = self._check_dict_key_tuples(flattened_hierarchy)
                self.hierarchy_depth[switch] = hierarchy_check[1]
                if not hierarchy_check[0]:
                    raise ValueError(
                        'Hierarchy mapping contains different levels for key "' + switch + '"'
                    )
                self.hierarchy[switch] = {
                    (k if isinstance(t, tuple) else t): v
                    for t, v in flattened_hierarchy.items()
                    for k in t
                }
        elif isinstance(hierarchy, pd.DataFrame):
            self.hierarchy = hierarchy
            self.hierarchy_depth = {}
            for col in self.cols:
                HIER_cols = self.hierarchy.columns[
                    self.hierarchy.columns.str.startswith(f'HIER_{col}')
                ].tolist()
                HIER_levels = [int(i.replace(f'HIER_{col}_', '')) for i in HIER_cols]
                if np.array_equal(sorted(HIER_levels), np.arange(1, max(HIER_levels) + 1)):
                    self.hierarchy_depth[col] = max(HIER_levels)
                else:
                    raise ValueError(f'Hierarchy columns are not complete for column {col}')
        elif hierarchy is None:
            self.hierarchy = hierarchy
        else:
            raise ValueError('Given hierarchy mapping is neither a dictionary nor a dataframe')

        self.cols_hier = []

    @staticmethod
    def _check_dict_key_tuples(dict_to_check: dict[Any, tuple]) -> tuple[bool, int]:
        """Check if all tuples in the dict values have the same length.

        Parameters
        ----------
        dict_to_check: dictionary to check

        Returns
        -------
        tuple: first entry if all sizes are equal, second minimum size.
        """
        min_tuple_size = min(len(v) for v in dict_to_check.values())
        max_tuple_size = max(len(v) for v in dict_to_check.values())
        return min_tuple_size == max_tuple_size, min_tuple_size

    def _fit(self, X: util.X_type, y: util.y_type, **kwargs) -> None:
        if isinstance(self.hierarchy, dict):
            X_hier = pd.DataFrame()
            for switch in self.hierarchy:
                if switch in self.cols:
                    colnames = [
                        f'HIER_{str(switch)}_{str(i + 1)}'
                        for i in range(self.hierarchy_depth[switch])
                    ]
                    df = pd.DataFrame(
                        X[str(switch)].map(self.hierarchy[str(switch)]).tolist(),
                        index=X.index,
                        columns=colnames,
                    )
                    X_hier = pd.concat([X_hier, df], axis=1)
        elif isinstance(self.hierarchy, pd.DataFrame):
            X_hier = self.hierarchy

        if isinstance(self.hierarchy, (dict, pd.DataFrame)):
            enc_hier = OrdinalEncoder(
                verbose=self.verbose,
                cols=X_hier.columns,
                handle_unknown='value',
                handle_missing='value',
            )
            enc_hier = enc_hier.fit(X_hier)
            X_hier_ordinal = enc_hier.transform(X_hier)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        if self.hierarchy is not None:
            self.mapping = self.fit_target_encoding(
                pd.concat([X_ordinal, X_hier_ordinal], axis=1), y
            )
        else:
            self.mapping = self.fit_target_encoding(X_ordinal, y)

    def fit_target_encoding(self, X: util.X_type, y: util.y_type) -> dict[str, np.ndarray]:
        """Fit the target encoding mapping.

        Parameters
        ----------
        X: training data to fit on.
        y: training target.

        Returns
        -------
        dictionary: column -> encoding values for column

        """
        mapping = {}
        prior = self._mean = y.mean()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            if 'HIER_' not in str(col):
                values = switch.get('mapping')

                scalar = prior
                if (isinstance(self.hierarchy, dict) and col in self.hierarchy) or (
                    isinstance(self.hierarchy, pd.DataFrame)
                ):
                    for i in range(self.hierarchy_depth[col]):
                        col_hier = 'HIER_' + str(col) + '_' + str(i + 1)
                        col_hier_m1 = (
                            col
                            if i == self.hierarchy_depth[col] - 1
                            else 'HIER_' + str(col) + '_' + str(i + 2)
                        )
                        if not X[col].equals(X[col_hier]) and len(X[col_hier].unique()) > 1:
                            stats_hier = y.groupby(X[col_hier]).agg(['count', 'mean'])
                            smoove_hier = self._weighting(stats_hier['count'])
                            scalar_hier = (
                                scalar * (1 - smoove_hier) + stats_hier['mean'] * smoove_hier
                            )
                            scalar_hier_long = X[[col_hier_m1, col_hier]].drop_duplicates()
                            scalar_hier_long.index = np.arange(1, scalar_hier_long.shape[0] + 1)
                            scalar = scalar_hier_long[col_hier].map(scalar_hier.to_dict())

                stats = y.groupby(X[col]).agg(['count', 'mean'])
                smoove = self._weighting(stats['count'])

                smoothing = scalar * (1 - smoove) + stats['mean'] * smoove

                if self.handle_unknown == 'return_nan':
                    smoothing.loc[-1] = np.nan
                elif self.handle_unknown == 'value':
                    smoothing.loc[-1] = prior

                if self.handle_missing == 'return_nan':
                    smoothing.loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == 'value':
                    smoothing.loc[-2] = prior

                mapping[col] = smoothing

        return mapping

    def _transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        # Now X is the correct dimensions it works with pre fitted ordinal encoder
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)
        return X

    def target_encode(self, X_in: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding via encoder mapping."""
        X = X_in.copy(deep=True)

        # Was not mapping extra columns as self.featuer_names_in did not include new column
        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X

    def _weighting(self, n: int) -> float:
        # monotonically increasing function of n bounded between 0 and 1
        # sigmoid in this case, using scipy.expit for numerical stability
        return expit((n - self.min_samples_leaf) / self.smoothing)

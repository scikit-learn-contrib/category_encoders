"""Target Encoder"""
import numpy as np
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
import warnings

__author__ = 'chappers'


class TargetEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """Target encoding for categorical features.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    For the case of categorical target: features are replaced with a blend of posterior probability of the target
    given particular categorical value and the prior probability of the target over all the training data.

    For the case of continuous target: features are replaced with a blend of the expected value of the target
    given particular categorical value and the expected value of the target over all the training data.

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
    min_samples_leaf: int
        For regularization the weighted average between category mean and global mean is taken. The weight is
        an S-shaped curve between 0 and 1 with the number of samples for a category on the x-axis.
        The curve reaches 0.5 at min_samples_leaf. (parameter k in the original paper)
    smoothing: float
        smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
        The value must be strictly bigger than 0. Higher values mean a flatter S-curve (see min_samples_leaf).
    hierarchy: dict
        a dictionary of columns to map into hierarchies.  Dictionary key(s) should be the column name from X
        which requires mapping.  For multiple hierarchical maps, this should be a dictionary of dictionaries.

    Examples
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = TargetEncoder(cols=['CHAS', 'RAD'], min_samples_leaf=20, smoothing=10).fit(X, y)
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

    >>> X, y = load_compass()
    >>> hierarchical_map = {'compass': {'N': ('N', 'NE'), 'S': ('S', 'SE'), 'W': 'W'}}
    >>> enc = TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=hierarchical_map, cols=['compass']).fit(X.loc[:,['compass']], y)
    >>> hierarchy_dataset = enc.transform(X.loc[:,['compass']])
    >>> print(hierarchy_dataset['compass'].values)
    [0.62263617 0.62263617 0.90382995 0.90382995 0.90382995 0.17660024
     0.17660024 0.46051953 0.46051953 0.46051953 0.46051953 0.40332791
     0.40332791 0.40332791 0.40332791 0.40332791]
    >>> X, y = load_postcodes('binary')
    >>> cols = ['postcode']
    >>> HIER_cols = ['HIER_postcode_1','HIER_postcode_2','HIER_postcode_3','HIER_postcode_4']
    >>> enc = TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2, hierarchy=X[HIER_cols], cols=['postcode']).fit(X['postcode'], y)
    >>> hierarchy_dataset = enc.transform(X['postcode'])
    >>> print(hierarchy_dataset.loc[0:10, 'postcode'].values)
    [0.75063473 0.90208756 0.88328833 0.77041254 0.68891504 0.85012847
    0.76772574 0.88742357 0.7933824  0.63776756 0.9019973 ]

    References
    ----------

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, from
    https://dl.acm.org/citation.cfm?id=507538

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value',
                 handle_unknown='value', min_samples_leaf=1, smoothing=1.0, hierarchy=None):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        if min_samples_leaf == 1:
            warnings.warn("Default parameter min_samples_leaf will change in version 2.6."
                          "See https://github.com/scikit-learn-contrib/category_encoders/issues/327",
                          category=FutureWarning)
        self.smoothing = smoothing
        if smoothing == 1.0:
            warnings.warn("Default parameter smoothing will change in version 2.6."
                          "See https://github.com/scikit-learn-contrib/category_encoders/issues/327",
                          category=FutureWarning)
        self.mapping = None
        self._mean = None
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
                    raise ValueError('Hierarchy mapping contains different levels for key "' + switch + '"')
                self.hierarchy[switch] = {(k if isinstance(t, tuple) else t): v for t, v in flattened_hierarchy.items() for k in t}
        elif isinstance(hierarchy, pd.DataFrame):
            self.hierarchy = hierarchy
            self.hierarchy_depth = {}
            for col in self.cols:
                HIER_cols = self.hierarchy.columns[self.hierarchy.columns.str.startswith(f'HIER_{col}')].values
                HIER_levels = [int(i.replace(f'HIER_{col}_', '')) for i in HIER_cols]
                if np.array_equal(sorted(HIER_levels), np.arange(1, max(HIER_levels)+1)):
                    self.hierarchy_depth[col] = max(HIER_levels)
                else:
                    raise ValueError(f'Hierarchy columns are not complete for column {col}')
        elif hierarchy is None:
            self.hierarchy = hierarchy
        else:
            raise ValueError('Given hierarchy mapping is neither a dictionary nor a dataframe')

        self.cols_hier = []

    def _check_dict_key_tuples(self, d):
        min_tuple_size = min(len(v) for v in d.values())
        max_tuple_size = max(len(v) for v in d.values())
        return min_tuple_size == max_tuple_size, min_tuple_size

    def _fit(self, X, y, **kwargs):
        if isinstance(self.hierarchy, dict):
            if not set(self.cols).issubset(X.columns):
                raise ValueError('X does not contain the columns listed in cols')
            X_hier = pd.DataFrame()
            for switch in self.hierarchy:
                if switch in self.cols:
                    colnames = [f'HIER_{str(switch)}_{str(i + 1)}' for i in range(self.hierarchy_depth[switch])]
                    df = pd.DataFrame(X[str(switch)].map(self.hierarchy[str(switch)]).tolist(), index=X.index, columns=colnames)
                    X_hier = pd.concat([X_hier, df], axis=1)
        elif isinstance(self.hierarchy, pd.DataFrame):
            X_hier = self.hierarchy

        if isinstance(self.hierarchy, (dict, pd.DataFrame)):
            enc_hier = OrdinalEncoder(
                verbose=self.verbose,
                cols=X_hier.columns,
                handle_unknown='value',
                handle_missing='value'
            )
            enc_hier = enc_hier.fit(X_hier)
            X_hier_ordinal = enc_hier.transform(X_hier)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        if self.hierarchy is not None:
            self.mapping = self.fit_target_encoding(pd.concat([X_ordinal, X_hier_ordinal], axis=1), y)
        else:
            self.mapping = self.fit_target_encoding(X_ordinal, y)

    def fit_target_encoding(self, X, y):
        mapping = {}
        prior = self._mean = y.mean()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            if 'HIER_' not in str(col):
                values = switch.get('mapping')
                
                scalar = prior
                if (isinstance(self.hierarchy, dict) and col in self.hierarchy) or \
                                    (isinstance(self.hierarchy, pd.DataFrame)):
                    for i in range(self.hierarchy_depth[col]):
                        col_hier = 'HIER_'+str(col)+'_'+str(i+1)
                        col_hier_m1 = col if i == self.hierarchy_depth[col]-1 else 'HIER_'+str(col)+'_'+str(i+2)
                        if not X[col].equals(X[col_hier]) and len(X[col_hier].unique())>1:
                            stats_hier = y.groupby(X[col_hier]).agg(['count', 'mean'])
                            smoove_hier = self._weighting(stats_hier['count'])
                            scalar_hier = scalar * (1 - smoove_hier) + stats_hier['mean'] * smoove_hier
                            scalar_hier_long = X[[col_hier_m1, col_hier]].drop_duplicates()
                            scalar_hier_long.index = np.arange(1, scalar_hier_long.shape[0]+1)
                            scalar = scalar_hier_long[col_hier].map(scalar_hier.to_dict())

                stats = y.groupby(X[col]).agg(['count', 'mean'])
                smoove = self._weighting(stats['count'])

                smoothing = scalar * (1 - smoove) + stats['mean'] * smoove
                smoothing[stats['count'] == 1] = scalar

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

    def _transform(self, X, y=None):
        # Now X is the correct dimensions it works with pre fitted ordinal encoder
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)
        return X

    def target_encode(self, X_in):
        X = X_in.copy(deep=True)

        # Was not mapping extra columns as self.cols did not include new column
        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X

    def _weighting(self, n):
        # monotonically increasing function on n bounded between 0 and 1
        return 1 / (1 + np.exp(-(n - self.min_samples_leaf) / self.smoothing))

"""Target Encoder"""
import numpy as np
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

    Example
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

    References
    ----------

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, from
    https://dl.acm.org/citation.cfm?id=507538

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value',
                 handle_unknown='value', min_samples_leaf=1, smoothing=1.0, heirarchy=None):
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
        self.heirarchy = heirarchy
        self.cols_heir = []

    def _fit(self, X, y, **kwargs):
        X = X.copy()
        if self.heirarchy:
            for switch in self.heirarchy:
                if switch in X.columns:
                    new_column = 'HEIR_'+switch
                    X[new_column] = X[switch].map(self.heirarchy[switch])
                    self.cols_heir.append(new_column)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols + self.cols_heir,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_target_encoding(X_ordinal, y)

    def fit_target_encoding(self, X, y):
        mapping = {}
        prior = self._mean = y.mean()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            if 'HEIR_' not in col:
                values = switch.get('mapping')

                scalar = prior
                if self.heirarchy and col in self.heirarchy:
                    col_heir = 'HEIR_'+col
                    stats_heir = y.groupby(X[col_heir]).agg(['count', 'mean'])
                    smoove_heir = self._weighting(stats_heir['count'])
                    scalar_heir = scalar * (1 - smoove_heir) + stats_heir['mean'] * smoove_heir
                    scalar_heir_long = X[[col, col_heir]].drop_duplicates()
                    scalar_heir_long.index = np.arange(1, scalar_heir_long.shape[0]+1)
                    scalar = scalar_heir_long[col_heir].map(scalar_heir.to_dict())
                    X.drop([col_heir], axis=1, inplace=True)
                    self.ordinal_encoder._dim -= 1
                    self.ordinal_encoder.cols.remove(col_heir)

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

        self.ordinal_encoder.mapping = [map for map in self.ordinal_encoder.mapping if not "HEIR_" in map['col']]

        return mapping

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)
        return X

    def target_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X

    def _weighting(self, n):
        # monotonically increasing function on n bounded between 0 and 1
        return 1 / (1 + np.exp(-(n - self.min_samples_leaf) / self.smoothing))

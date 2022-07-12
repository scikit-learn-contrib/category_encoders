"""Target Encoder"""
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder
import category_encoders.utils as util
import pandas as pd
import warnings

__author__ = 'chappers, nercisla'


class TargetEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """Target encoding for categorical features.

    Supported targets: binomial, multi-class and continuous. For polynomial target support, see PolynomialWrapper.

    For the case of categorical target: features are replaced with a blend of posterior probability of the target
    given particular categorical value and the prior probability of the target over all the training data.  Multiclass
    targets are allowed.

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
                 handle_unknown='value', min_samples_leaf=1, smoothing=1.0, multiclass_target=False):
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
        self.y_colnames = None
        self.X_colnames = None
        self.multiclass_target = multiclass_target

    def _fit(self, X, y, **kwargs):

        if len(y.unique()) > 2 and self.multiclass_target == True:
            warnings.warn("The target is multiclass and will be one hot encoded.", category=UserWarning)
            ohe_encoder = OneHotEncoder(
                verbose=self.verbose,
                handle_unknown='error',
                handle_missing='error'
            )
            ohe_encoder = ohe_encoder.fit(y.astype(str))
            y = ohe_encoder.transform(y.astype(str))
            self.y_colnames = y.columns
            self.X_colnames = [col + "_" + ycol for ycol in y.columns for col in self.cols]
        else:
            warnings.warn("The target is multiclass but multiclass_target == False.", category=UserWarning)
        
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        if self.multiclass_target == False:
            self.y_colnames = ['y']
            self.X_colnames = self.cols
        self.mapping = self.fit_target_encoding(X_ordinal, y)

    def fit_target_encoding(self, X, y):
        mapping = {}

        for switch in self.ordinal_encoder.category_mapping:

            col = switch.get('col')
            values = switch.get('mapping')

            y = pd.DataFrame({'target': y}) if len(y.shape) == 1 else y
            for i in range(y.shape[1]):
                yi = y.iloc[:, i]
                ycol = None if y.shape[1] == 1 else y.columns[i]

                prior = self._mean = yi.mean()

                stats = yi.groupby(X[col]).agg(['count', 'mean'])

                smoove = 1 / (1 + np.exp(-(stats['count'] - self.min_samples_leaf) / self.smoothing))
                smoothing = prior * (1 - smoove) + stats['mean'] * smoove
                smoothing[stats['count'] == 1] = prior

                if self.handle_unknown == 'return_nan':
                    smoothing.loc[-1] = np.nan
                elif self.handle_unknown == 'value':
                    smoothing.loc[-1] = prior

                if self.handle_missing == 'return_nan':
                    smoothing.loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == 'value':
                    smoothing.loc[-2] = prior

                colname = col+"_"+ycol if ycol is not None else col
                mapping[colname] = smoothing

        return mapping

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)
        
        X_rep = pd.concat([X[self.cols]]*len(self.y_colnames),axis=1)
        X_rep.columns = self.X_colnames
        X = pd.concat([X_rep, X[X.columns.difference(self.cols, sort=False)]], axis=1)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)
        return X

    def target_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.X_colnames:
            X[col] = X[col].map(self.mapping[col])

        return X

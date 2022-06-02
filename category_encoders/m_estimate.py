"""M-probability estimate"""
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class MEstimateEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """M-probability estimate of likelihood.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    This is a simplified version of target encoder, which goes under names like m-probability estimate or
    additive smoothing with known incidence rates. In comparison to target encoder, m-probability estimate
    has only one tunable parameter (`m`), while target encoder has two tunable parameters (`min_samples_leaf`
    and `smoothing`).

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop encoded columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    m: float
        this is the "m" in the m-probability estimate. Higher value of m results into stronger shrinking.
        M is non-negative.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = MEstimateEncoder(cols=['CHAS', 'RAD']).fit(X, y)
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

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, equation 7, from
    https://dl.acm.org/citation.cfm?id=507538

    .. [2] On estimating probabilities in tree pruning, equation 1, from
    https://link.springer.com/chapter/10.1007/BFb0017010

    .. [3] Additive smoothing, from
    https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, m=1.0):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self.mapping = None
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.m = m

    def _fit(self, X, y, **kwargs):

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X, y)
        return X

    def _more_tags(self):
        tags = super()._more_tags()
        tags["predict_depends_on_y"] = True
        return tags

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()
        prior = self._sum/self._count

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['sum', 'count'])  # Count of x_{i,+} and x_i

            # Calculate the m-probability estimate
            estimate = (stats['sum'] + prior * self.m) / (stats['count'] + self.m)

            # Ignore unique columns. This helps to prevent overfitting on id-like columns
            if len(stats['count']) == self._count:
                estimate[:] = prior

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = prior

            # Store the m-probability estimate for transform() function
            mapping[col] = estimate

        return mapping

    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X

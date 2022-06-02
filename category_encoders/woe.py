"""Weight of Evidence"""
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class WOEEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
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
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    regularization: float
        the purpose of regularization is mostly to prevent division by zero.
        When regularization is 0, you may encounter division by zero.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = WOEEncoder(cols=['CHAS', 'RAD']).fit(X, y)
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

    .. [1] Weight of Evidence (WOE) and Information Value Explained, from
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, regularization=1.0):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization

    def _fit(self, X, y, **kwargs):
        # The label must be binary with values {0,1}
        unique = y.unique()
        if len(unique) != 2:
            raise ValueError("The target column y must be binary. But the target contains " + str(len(unique)) + " unique value(s).")
        if y.isnull().any():
            raise ValueError("The target column y must not contain missing values.")
        if np.max(unique) < 1:
            raise ValueError("The target column y must be binary with values {0, 1}. Value 1 was not found in the target.")
        if np.min(unique) > 0:
            raise ValueError("The target column y must be binary with values {0, 1}. Value 0 was not found in the target.")

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
            nominator = (stats['sum'] + self.regularization) / (self._sum + 2*self.regularization)
            denominator = ((stats['count'] - stats['sum']) + self.regularization) / (self._count - self._sum + 2*self.regularization)
            woe = np.log(nominator / denominator)

            # Ignore unique values. This helps to prevent overfitting on id-like columns.
            woe[stats['count'] == 1] = 0

            if self.handle_unknown == 'return_nan':
                woe.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                woe.loc[-1] = 0

            if self.handle_missing == 'return_nan':
                woe.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                woe.loc[-2] = 0

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
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X

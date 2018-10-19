"""Weight of Evidence"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import get_obj_cols, convert_input, get_generated_cols
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence coding for categorical features.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'ignore', 'error' and 'impute', defaults to 'impute', which will assume WOE=0.
        Values that are observed only once during training are treated as if they were not observed at all.
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

    .. [1] Weight of Evidence (WOE) and Information Value Explained. from
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html.


    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True,
                 handle_unknown='impute', random_state=None, randomized=False, sigma=0.05, regularization=1.0):
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization

    # noinspection PyUnusedLocal
    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and binary y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Binary target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # Unite parameters into pandas types
        X = convert_input(X)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        else:
            y = pd.Series(y, name='target')

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        # The label must be binary with values {0,1}
        unique = y.unique()
        if len(unique) != 2:
            raise ValueError("The target column y must be binary. But the target contains " +
                             str(len(unique)) + " unique value(s).")
        if y.isnull().any():
            raise ValueError(
                "The target column y must not contain missing values.")
        if np.max(unique) < 1:
            raise ValueError(
                "The target column y must be binary with values {0, 1}. Value 1 was not found in the target.")
        if np.min(unique) > 0:
            raise ValueError(
                "The target column y must be binary with values {0, 1}. Value 0 was not found in the target.")

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        # Training
        self.mapping = self._train(X, y, cols=self.cols)

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X, y=None):
        """Perform the transformation to new categorical data. When the data are used for model training,
        it is important to also pass the target in order to apply leave one out.

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

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # Unite the input into pandas DataFrame
        X = convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim,))

        # If we are encoding the training data, we have to check the target
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            else:
                y = pd.Series(y, name='target')
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not self.cols:
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
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
        return self.fit(X, y, **fit_params).transform(X, y)

    def _train(self, X, y, cols=None):
        # Initialize the output
        mapping_out = []

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()

        for col in cols:
            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(
                ['sum', 'count'])  # Count of x_{i,+} and x_i
            stats = stats.to_dict(orient='index')

            # Initialization
            woe = {}

            # Create a new column with regularized WOE.
            # Regularization helps to avoid division by zero.
            for val in stats:
                # Pre-calculate WOEs because logarithms are slow.
                # Ignore unique values. This helps to prevent overfitting on id-like columns and keep the model small.
                if stats[val]['count'] > 1:
                    nominator = (
                        stats[val]['sum'] + self.regularization) / (self._sum + 2*self.regularization)
                    denominator = ((stats[val]['count'] - stats[val]['sum']) + self.regularization) / (
                        self._count - self._sum + 2*self.regularization)
                    woe[val] = np.log(nominator / denominator)

            # Store the column statistics for transform() function
            mapping_out.append({'col': col, 'woe': woe})

        return mapping_out

    def _score(self, X, y):
        for switch in self.mapping:
            # Get column name (can be anything: str, number,...)
            column = switch.get('col')

            # Score the column
            transformed_column = pd.Series([np.nan] * X.shape[0], name=column)
            for val in switch.get('woe'):
                transformed_column.loc[X[column] == val] = switch.get(
                    'woe')[val]  # THIS LINE IS SLOW

            # Replace missing values only in the computed columns
            if self.impute_missing:
                if self.handle_unknown == 'impute':
                    transformed_column.fillna(0, inplace=True)
                elif self.handle_unknown == 'error':
                    missing = transformed_column.isnull()
                    if any(missing):
                        raise ValueError(
                            'Unexpected categories found in column %s' % switch.get('col'))

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                transformed_column = (transformed_column * random_state_generator.normal(
                    1., self.sigma, transformed_column.shape[0]))

            X[column] = transformed_column.astype(float)
        return X

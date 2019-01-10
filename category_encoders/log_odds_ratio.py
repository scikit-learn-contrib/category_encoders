"""Log-odds ratio"""
import scipy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state
from scipy import optimize

__author__ = 'Jan Motl'

class LogOddsRatioEncoder(BaseEstimator, TransformerMixin):
    """Log-odds ratio as estimated with empirical Bayes.

    The advantage of empirical Bayes, in contrast to target encoding, is that there is no parameter to fine-tune.


    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop encoded columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0 (the expected log-odds ratio).
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0 (the expected log-odds ratio).
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.


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

    .. [1] Shrinkage Estimation of Log-odds Ratios for Comparing Mobility Tables, from
    https://journals.sagepub.com/doi/abs/10.1177/0081175015570097


    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, regularization=1.0):
        self.verbose = verbose
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization
        self.feature_names = None

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
        X = util.convert_input(X)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:,0]
        else:
            y = pd.Series(y, name='target', index=X.index)

        # The lengths must be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

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

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

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

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # Store column names with approximately constant variance on the training data
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))
        return self

    def transform(self, X, y=None, override_return_df=False):
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

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # Unite the input into pandas DataFrame
        X = util.convert_input(X)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # If we are encoding the training data, we have to check the target
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            else:
                y = pd.Series(y, name='target', index=X.index)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not self.cols:
            return X

        # Do not modify the input argument
        X = X.copy(deep=True)

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
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

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()

        # Iterative estimation of mu and sigma as given on page 9.
        # This problem is traditionally solved with Newton–Raphson method:
        #   https://en.wikipedia.org/wiki/Newton%27s_method
        # But we just use sklearn minimizer.
        def get_best_sigma(sigma, mu_k, sigma_k, K):
            global mu                               # Ugly. But I want to be able to read it once the optimization ends.
            w_k = 1. / (sigma ** 2 + sigma_k ** 2)  # Weights depends on sigma
            mu = sum(w_k * mu_k) / sum(w_k)         # Mu transitively depends on sigma
            total = sum(w_k * (mu_k - mu) ** 2)     # We want this to be close to K-1
            loss = abs(total - (K - 1))
            return loss

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['sum', 'count']) # Count of x_{i,+} and x_i

            # Create 2x2 contingency table
            crosstable = pd.DataFrame()
            crosstable['E-A-'] = self._count - stats['count'] + stats['sum'] - self._sum
            crosstable['E-A+'] = stats['count'] - stats['sum']
            crosstable['E+A-'] = self._sum - stats['sum']
            crosstable['E+A+'] = stats['sum']
            index = crosstable.index.values
            crosstable = np.array(crosstable, dtype=np.float32) # The argument unites the types into float

            # Count of contingency tables.
            K = len(crosstable)

            if K>1: # We want to avoid division by zero in y_k calculation
                # Estimate log-odds ratios with Yates correction as listed on page 5.
                mu_k = np.log((crosstable[:, 0] + 0.5) * (crosstable[:, 3] + 0.5) / ((crosstable[:, 1] + 0.5) * (crosstable[:, 2] + 0.5)))

                # Standard deviation estimate for 2x2 contingency table as given in equation 2.
                # The explanation of the equation is given in:
                #   https://stats.stackexchange.com/questions/266098/how-do-i-calculate-the-standard-deviation-of-the-log-odds
                sigma_k = np.sqrt(np.sum(1. / (crosstable + 0.5), axis=1))

                # Estimate the sigma and mu. Sigma is non-negative.
                result = scipy.optimize.minimize(get_best_sigma, x0=np.mean(sigma_k), args=(mu_k, sigma_k, K), bounds=[(0, np.inf)])
                sigma = result.x[0]

                # Empirical Bayes follows equation 7.
                # However, James-Stein estimator behaves perversely when K < 3. Hence, we clip the B into interval <0,1>.
                B = (K - 3) * sigma_k ** 2 / ((K - 1) * (sigma ** 2 + sigma_k ** 2))
                B = B.clip(0,1)
                y_k = mu + (1 - B) * (mu_k - mu)

                # Convert Numpy vector back into Series
                estimate = pd.Series(y_k, index=index)
            else:
                estimate = pd.Series(0, index=values)

            # Ignore unique values. This helps to prevent overfitting on id-like columns
            # estimate[stats['count'] == 1] = 0

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = 0

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = 0

            # Store the estimate for transform() function
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

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns:
        --------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names

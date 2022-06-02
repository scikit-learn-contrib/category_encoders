"""James-Stein"""
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Jan Motl'


class JamesSteinEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """James-Stein estimator.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    For feature value `i`, James-Stein estimator returns a weighted average of:

        1. The mean target value for the observed feature value `i`.
        2. The mean target value (regardless of the feature value).

    This can be written as::

        JS_i = (1-B)*mean(y_i) + B*mean(y)

    The question is, what should be the weight `B`?
    If we put too much weight on the conditional mean value, we will overfit.
    If we put too much weight on the global mean, we will underfit.
    The canonical solution in machine learning is to perform cross-validation.
    However, Charles Stein came with a closed-form solution to the problem.
    The intuition is: If the estimate of `mean(y_i)` is unreliable (`y_i` has high variance),
    we should put more weight on `mean(y)`. Stein put it into an equation as::

        B = var(y_i) / (var(y_i)+var(y))

    The only remaining issue is that we do not know `var(y)`, let alone `var(y_i)`.
    Hence, we have to estimate the variances. But how can we reliably estimate the
    variances, when we already struggle with the estimation of the mean values?!
    There are multiple solutions:

        1. If we have the same count of observations for each feature value `i` and all
        `y_i` are close to each other, we can pretend that all `var(y_i)` are identical.
        This is called a pooled model.
        2. If the observation counts are not equal, it makes sense to replace the variances
        with squared standard errors, which penalize small observation counts::

            SE^2 = var(y)/count(y)

        This is called an independent model.

    James-Stein estimator has, however, one practical limitation - it was defined
    only for normal distributions. If you want to apply it for binary classification,
    which allows only values {0, 1}, it is better to first convert the mean target value
    from the bound interval <0,1> into an unbounded interval by replacing mean(y)
    with log-odds ratio::

        log-odds_ratio_i = log(mean(y_i)/mean(y_not_i))

    This is called binary model. The estimation of parameters of this model is, however,
    tricky and sometimes it fails fatally. In these situations, it is better to use beta
    model, which generally delivers slightly worse accuracy than binary model but does
    not suffer from fatal failures.

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
    model: str
        options are 'pooled', 'beta', 'binary' and 'independent', defaults to 'independent'.
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
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = JamesSteinEncoder(cols=['CHAS', 'RAD']).fit(X, y)
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

    .. [1] Parametric empirical Bayes inference: Theory and applications, equations 1.19 & 1.20, from
    https://www.jstor.org/stable/2287098

    .. [2] Empirical Bayes for multiple sample sizes, from
    http://chris-said.io/2017/05/03/empirical-bayes-for-multiple-sample-sizes/

    .. [3] Shrinkage Estimation of Log-odds Ratios for Comparing Mobility Tables, from
    https://journals.sagepub.com/doi/abs/10.1177/0081175015570097

    .. [4] Stein's paradox and group rationality, from
    http://www.philos.rug.nl/~romeyn/presentation/2017_romeijn_-_Paris_Stein.pdf

    .. [5] Stein's Paradox in Statistics, from
    http://statweb.stanford.edu/~ckirby/brad/other/Article1977.pdf

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value',
                 handle_missing='value', model='independent', random_state=None, randomized=False, sigma=0.05):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self.mapping = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.model = model

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
        if self.model == 'independent':
            self.mapping = self._train_independent(X_ordinal, y)
        elif self.model == 'pooled':
            self.mapping = self._train_pooled(X_ordinal, y)
        elif self.model == 'beta':
            self.mapping = self._train_beta(X_ordinal, y)
        elif self.model == 'binary':
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
            # Perform the training
            self.mapping = self._train_log_odds_ratio(X_ordinal, y)
        else:
            raise ValueError("model='" + str(self.model) + "' is not a recognized option")

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)
        return X

    def _more_tags(self):
        tags = super()._more_tags()
        tags["predict_depends_on_y"] = True
        return tags

    def _train_pooled(self, X, y):
        # Implemented based on reference [1]

        # Initialize the output
        mapping = {}

        # Calculate global statistics
        prior = y.mean()
        target_var = y.var()
        global_count = len(y)

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['mean', 'count'])

            # See: Computer Age Statistical Inference: Algorithms, Evidence, and Data Science (Bradley Efron & Trevor Hastie, 2016)
            #   Equations 7.19 and 7.20.
            # Note: The equations assume normal distribution of the label. But our label is p(y|x),
            # which is definitely not normally distributed as probabilities are bound to lie on interval 0..1.
            # We make this approximation because Efron does it as well.

            # Equation 7.19
            # Explanation of the equation:
            #   https://stats.stackexchange.com/questions/191444/variance-in-estimating-p-for-a-binomial-distribution
            # if stats['count'].var() > 0:
            #     warnings.warn('The pooled model assumes that each category is observed exactly N times. This was violated in "' + str(col) +'" column. Consider comparing the accuracy of this model to "independent" model.')
            # This is a parametric estimate of var(p) in the binomial distribution.
            # We do not use it because we also want to support non-binary targets.
            # The difference in the estimates is small.
            #   variance = prior * (1 - prior) / stats['count'].mean()
            # This is a squared estimate of standard error of the mean:
            #   https://en.wikipedia.org/wiki/Standard_error
            variance = target_var/(stats['count'].mean())

            # Equation 7.20
            SSE = ((stats['mean']-prior)**2).sum()  # Sum of Squared Errors
            if SSE > 0:  # We have to avoid division by zero
                B = ((len(stats['count'])-3)*variance) / SSE
                B = B.clip(0, 1)
                estimate = prior + (1 - B) * (stats['mean'] - prior)
            else:
                estimate = stats['mean']

            # Ignore unique values. This helps to prevent overfitting on id-like columns
            # This works better than: estimate[stats['count'] == 1] = prior
            if len(stats['mean']) == global_count:
                estimate[:] = prior

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = prior

            # Store the estimate for transform() function
            mapping[col] = estimate

        return mapping

    def _train_independent(self, X, y):
        # Implemented based on reference [2]

        # Initialize the output
        mapping = {}

        # Calculate global statistics
        prior = y.mean()
        global_count = len(y)
        global_var = y.var()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['mean', 'var'])

            i_var = stats['var'].fillna(0)   # When we do not have more than 1 sample, assume 0 variance
            unique_cnt = len(X[col].unique())

            # See: Parametric Empirical Bayes Inference: Theory and Applications (Morris, 1983)
            #   Equations 1.19 and 1.20.
            # Note: The equations assume normal distribution of the label. But our label is p(y|x),
            # which is definitely not normally distributed as probabilities are bound to lie on interval 0..1.
            # Nevertheless, it seems to perform surprisingly well. This is in agreement with:
            #   Data Analysis with Stein's Estimator and Its Generalizations (Efron & Morris, 1975)
            # The equations are similar to James-Stein estimator, as listed in:
            #   Stein's Paradox in Statistics (Efron & Morris, 1977)
            # Or:
            #   Computer Age Statistical Inference: Algorithms, Evidence, and Data Science (Efron & Hastie, 2016)
            #   Equations 7.19 and 7.20.
            # The difference is that they have equal count of observations per estimated variable, while we generally
            # do not have that. Nice discussion about that is given at:
            #   http://chris-said.io/2017/05/03/empirical-bayes-for-multiple-sample-sizes/
            smoothing = i_var / (global_var + i_var) * (unique_cnt-3) / (unique_cnt-1)
            smoothing = 1 - smoothing
            smoothing = smoothing.clip(lower=0, upper=1)   # Smoothing should be in the interval <0,1>

            estimate = smoothing*(stats['mean']) + (1-smoothing)*prior

            # Ignore unique values. This helps to prevent overfitting on id-like columns
            if len(stats['mean']) == global_count:
                estimate[:] = prior

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = prior

            # Store the estimate for transform() function
            mapping[col] = estimate

        return mapping

    def _train_log_odds_ratio(self, X, y):
        # Implemented based on reference [3]

        # Initialize the output
        mapping = {}

        # Calculate global statistics
        global_sum = y.sum()
        global_count = y.count()

        # Iterative estimation of mu and sigma as given on page 9.
        # This problem is traditionally solved with Newton-Raphson method:
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
            stats = y.groupby(X[col]).agg(['sum', 'count'])  # Count of x_{i,+} and x_i

            # Create 2x2 contingency table
            crosstable = pd.DataFrame()
            crosstable['E-A-'] = global_count - stats['count'] + stats['sum'] - global_sum
            crosstable['E-A+'] = stats['count'] - stats['sum']
            crosstable['E+A-'] = global_sum - stats['sum']
            crosstable['E+A+'] = stats['sum']
            index = crosstable.index.values
            crosstable = np.array(crosstable, dtype=np.float32)  # The argument unites the types into float

            # Count of contingency tables.
            K = len(crosstable)

            # Ignore id-like columns. This helps to prevent overfitting.
            if K == global_count:
                estimate = pd.Series(0, index=values)
            else:
                if K > 1:  # We want to avoid division by zero in y_k calculation
                    # Estimate log-odds ratios with Yates correction as listed on page 5.
                    mu_k = np.log((crosstable[:, 0] + 0.5) * (crosstable[:, 3] + 0.5) / ((crosstable[:, 1] + 0.5) * (crosstable[:, 2] + 0.5)))

                    # Standard deviation estimate for 2x2 contingency table as given in equation 2.
                    # The explanation of the equation is given in:
                    #   https://stats.stackexchange.com/questions/266098/how-do-i-calculate-the-standard-deviation-of-the-log-odds
                    sigma_k = np.sqrt(np.sum(1. / (crosstable + 0.5), axis=1))

                    # Estimate the sigma and mu. Sigma is non-negative.
                    result = scipy.optimize.minimize(get_best_sigma, x0=1e-4, args=(mu_k, sigma_k, K), bounds=[(0, np.inf)], method='TNC', tol=1e-12, options={'gtol': 1e-12, 'ftol': 1e-12, 'eps': 1e-12})
                    sigma = result.x[0]

                    # Empirical Bayes follows equation 7.
                    # However, James-Stein estimator behaves perversely when K < 3. Hence, we clip the B into interval <0,1>.
                    # Literature reference for the clipping:
                    #   Estimates of Income for Small Places: An Application of James-Stein Procedures to Census Data (Fay & Harriout, 1979),
                    # page 270.
                    B = (K - 3) * sigma_k ** 2 / ((K - 1) * (sigma ** 2 + sigma_k ** 2))
                    B = B.clip(0, 1)
                    y_k = mu + (1 - B) * (mu_k - mu)

                    # Convert Numpy vector back into Series
                    estimate = pd.Series(y_k, index=index)
                else:
                    estimate = pd.Series(0, index=values)

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

    def _train_beta(self, X, y):
        # Implemented based on reference [4]

        # Initialize the output
        mapping = {}

        # Calculate global statistics
        prior = y.mean()
        global_count = len(y)

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['mean', 'count'])

            # See: Stein's paradox and group rationality (Romeijn, 2017), page 14
            smoothing = stats['count'] / (stats['count'] + global_count)

            estimate = smoothing*(stats['mean']) + (1-smoothing)*prior

            # Ignore unique values. This helps to prevent overfitting on id-like columns
            if len(stats['mean']) == global_count:
                estimate[:] = prior

            if self.handle_unknown == 'return_nan':
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                estimate.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                estimate.loc[-2] = prior

            # Store the estimate for transform() function
            mapping[col] = estimate

        return mapping

    # todo this score function is copied 4 times
    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X

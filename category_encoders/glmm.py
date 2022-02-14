"""Generalized linear mixed model"""
import warnings
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.random import check_random_state
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM as bgmm

__author__ = 'Jan Motl'


class GLMMEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Generalized linear mixed model.

    Supported targets: binomial and continuous. For polynomial target support, see PolynomialWrapper.

    This is a supervised encoder similar to TargetEncoder or MEstimateEncoder, but there are some advantages:
    1) Solid statistical theory behind the technique. Mixed effects models are a mature branch of statistics.
    2) No hyper-parameters to tune. The amount of shrinkage is automatically determined through the estimation process.
    In short, the less observations a category has and/or the more the outcome varies for a category
    then the higher the regularization towards "the prior" or "grand mean".
    3) The technique is applicable for both continuous and binomial targets. If the target is continuous,
    the encoder returns regularized difference of the observation's category from the global mean.
    If the target is binomial, the encoder returns regularized log odds per category.

    In comparison to JamesSteinEstimator, this encoder utilizes generalized linear mixed models from statsmodels library.

    Note: This is an alpha implementation. The API of the method may change in the future.

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
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which returns 0.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    binomial_target: bool
        if True, the target must be binomial with values {0, 1} and Binomial mixed model is used.
        If False, the target must be continuous and Linear mixed model is used.
        If None (the default), a heuristic is applied to estimate the target type.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target > 22.5
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = GLMMEncoder(cols=['CHAS', 'RAD']).fit(X, y)
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

    .. [1] Data Analysis Using Regression and Multilevel/Hierarchical Models, page 253, from
    https://faculty.psau.edu.sa/filedownload/doc-12-pdf-a1997d0d31f84d13c1cdc44ac39a8f2c-original.pdf

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, binomial_target=None):
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
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.binomial_target = binomial_target
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
        X, y = util.convert_inputs(X, y)
        y = y.astype(float)

        self._dim = X.shape[1]

        # If columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
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
        """Perform the transformation to new categorical data.

        When the data are used for model training, it is important to also pass the target in order to apply leave one out.

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
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # Unite the input into pandas types
        X, y = util.convert_inputs(X, y, deep=True)

        # Then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not list(self.cols):
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over the columns and replace the nominal values with the numbers
        X = self._score(X, y)

        # Postprocessing
        # Note: We should not even convert these columns.
        if self.drop_invariant:
            X = X.drop(columns=self.drop_cols)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Estimate target type, if necessary
        if self.binomial_target is None:
            if len(y.unique()) <= 2:
                binomial_target = True
            else:
                binomial_target = False
        else:
            binomial_target = self.binomial_target

        # The estimation does not have to converge -> at least converge to the same value.
        original_state = np.random.get_state()
        np.random.seed(2001)

        # Reset random state on completion
        try:
            for switch in self.ordinal_encoder.category_mapping:
                col = switch.get('col')
                values = switch.get('mapping')
                data = self._rename_and_merge(X, y, col)

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        if binomial_target:
                            # Classification, returns (regularized) log odds per category as stored in vc_mean
                            # Note: md.predict() returns: output = fe_mean + vcp_mean + vc_mean[category]
                            md = bgmm.from_formula('target ~ 1', {'a': '0 + C(feature)'}, data).fit_vb()
                            index_names = [int(float(re.sub(r'C\(feature\)\[(\S+)\]', r'\1', index_name))) for index_name in md.model.vc_names]
                            estimate = pd.Series(md.vc_mean, index=index_names)
                        else:
                            # Regression, returns (regularized) mean deviation of the observation's category from the global mean
                            md = smf.mixedlm('target ~ 1', data, groups=data['feature']).fit()
                            tmp = dict()
                            for key, value in md.random_effects.items():
                                tmp[key] = value[0]
                            estimate = pd.Series(tmp)
                except np.linalg.LinAlgError:
                    # Singular matrix -> just return all zeros
                    estimate = pd.Series(np.zeros(len(values)), index=values)

                # Ignore unique columns. This helps to prevent overfitting on id-like columns
                if len(X[col].unique()) == len(y):
                    estimate[:] = 0

                if self.handle_unknown == 'return_nan':
                    estimate.loc[-1] = np.nan
                elif self.handle_unknown == 'value':
                    estimate.loc[-1] = 0

                if self.handle_missing == 'return_nan':
                    estimate.loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == 'value':
                    estimate.loc[-2] = 0

                mapping[col] = estimate
        finally:
            np.random.set_state(original_state)

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

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names

    def _rename_and_merge(self, X, y, col):
        """
        Statsmodels requires:
            1) unique column names
            2) non-numeric columns names
        Solution: internally rename the columns.
        """
        merged = pd.DataFrame()
        merged['feature'] = X[col]
        merged['target'] = y

        return merged

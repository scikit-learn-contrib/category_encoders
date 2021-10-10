"""Target Encoder"""
__author__ = "david26694", "cmougan"

import numpy as np
from sklearn.base import BaseEstimator
import category_encoders.utils as util
from sklearn.utils.random import check_random_state
class QuantileEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Quantile Encoding for categorical features.
    This a statistically modified version of target MEstimate encoder where selected features
    are replaced the statistical quantile instead than the mean. Replacing with the
    median is a particular case where self.quantile = 0.5. In comparison to MEstimateEncoder
    it has two tunable parameter `m` and `quantile`
    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    quantile: int
        integer indicating statistical quantile. ´0.5´ for median.
    m: int
        integer indicating the smoothing parameter. 0 for no smoothing.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target quantile.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target quantile.
    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = QuantileEncoder(cols=['CHAS', 'RAD']).fit(X, y)
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
    .. [1] Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, https://arxiv.org/abs/2105.13783
    .. [2] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, equation 7, from https://dl.acm.org/citation.cfm?id=507538
    .. [3] On estimating probabilities in tree pruning, equation 1, from https://link.springer.com/chapter/10.1007/BFb0017010
    .. [4] Additive smoothing, from https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates
    .. [5] Target encoding done the right way https://maxhalford.github.io/blog/target-encoding/
    """

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_missing="value",
        handle_unknown="value",
        quantile=0.5,
        m=1.0,
    ):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.feature_names = None
        self.quantile = quantile
        self.m = m

    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        # unite the input into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The length of X is "
                + str(X.shape[0])
                + " but length of y is "
                + str(y.shape[0])
                + "."
            )

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError("Columns to be encoded can not contain null")

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown="value",
            handle_missing="value",
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_quantile_encoding(X_ordinal, y)

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print(
                        "Could not remove column from feature names."
                        "Not found in generated cols.\n{}".format(e)
                    )

        return self

    def fit_quantile_encoding(self, X, y):
        mapping = {}

        # Calculate global statistics
        prior = self._quantile = np.quantile(y, self.quantile)
        self._sum = y.sum()
        self._count = y.count()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get("col")
            values = switch.get("mapping")

            # Calculate sum, count and quantile of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(
                [lambda x: np.quantile(x, self.quantile), "sum", "count"]
            )
            stats.columns = ["quantile", "sum", "count"]

            # Calculate the m-probability estimate of the quantile
            estimate = (stats["count"] * stats["quantile"] + prior * self.m) / (
                stats["count"] + self.m
            )

            if self.handle_unknown == "return_nan":
                estimate.loc[-1] = np.nan
            elif self.handle_unknown == "value":
                estimate.loc[-1] = prior

            if self.handle_missing == "return_nan":
                estimate.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == "value":
                estimate.loc[-2] = prior

            mapping[col] = estimate

        return mapping

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target info (such as transform test set)
        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.
        """

        if self.handle_missing == "error":
            if X[self.cols].isnull().any().any():
                raise ValueError("Columns to be encoded can not contain null")

        if self._dim is None:
            raise ValueError(
                "Must train encoder before it can be used to transform data."
            )

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError(
                "Unexpected input dimension %d, expected %d" % (X.shape[1], self._dim,)
            )

        # if we are encoding the training data, we have to check the target
        if y is not None:
            y = util.convert_input_vector(y, X.index)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "The length of X is "
                    + str(X.shape[0])
                    + " but length of y is "
                    + str(y.shape[0])
                    + "."
                )

        if not list(self.cols):
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError("Unexpected categories found in dataframe")

        X = self.quantile_encode(X)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def quantile_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

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
            raise ValueError(
                "Must fit data first. Affected feature names are not known " "before."
            )
        else:
            return self.feature_names

"""Quantile Encoder"""
__author__ = "david26694", "cmougan"

from functools import reduce
import operator
from typing import List
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder


class QuantileEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """Quantile Encoding for categorical features.

    This a statistically modified version of target MEstimate encoder where selected features
    are replaced by the statistical quantile instead of the mean. Replacing with the
    median is a particular case where self.quantile = 0.5. In comparison to MEstimateEncoder
    it has two tunable parameter `m` and `quantile`

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    quantile: float
        float indicating statistical quantile. ´0.5´ for median.
    m: float
        this is the “m” in the m-probability estimate. Higher value of m results into stronger shrinking. M is non-negative. 0 for no smoothing.
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
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = QuantileEncoder(cols=["CentralAir", "Heating"], quantile=0.5, m=1.0).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Id           1460 non-null   float64
     1   MSSubClass   1460 non-null   float64
     2   MSZoning     1460 non-null   object 
     3   LotFrontage  1201 non-null   float64
     4   YearBuilt    1460 non-null   float64
     5   Heating      1460 non-null   float64
     6   CentralAir   1460 non-null   float64
    dtypes: float64(6), object(1)
    memory usage: 80.0+ KB
    None

    References
    ----------

    .. [1] Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, https://link.springer.com/chapter/10.1007%2F978-3-030-85529-1_14
    .. [2] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, equation 7, from https://dl.acm.org/citation.cfm?id=507538
    .. [3] On estimating probabilities in tree pruning, equation 1, from https://link.springer.com/chapter/10.1007/BFb0017010
    .. [4] Additive smoothing, from https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates
    .. [5] Target encoding done the right way https://maxhalford.github.io/blog/target-encoding/
    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

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
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self.mapping = None
        self.quantile = quantile
        self.m = m

    def _fit(self, X, y, **kwargs):
        y = y.astype(float)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown="value",
            handle_missing="value",
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_quantile_encoding(X_ordinal, y)

    def fit_quantile_encoding(self, X, y):
        mapping = {}

        # Calculate global statistics
        prior = np.quantile(y, self.quantile)

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get("col")
            values = switch.get("mapping")

            # Calculate sum, count and quantile of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg([lambda x: np.quantile(x, self.quantile), "sum", "count"])
            stats.columns = ["quantile", "sum", "count"]

            # Calculate the m-probability estimate of the quantile
            estimate = (stats["count"] * stats["quantile"] + prior * self.m) / (stats["count"] + self.m)

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

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError("Unexpected categories found in dataframe")

        X = self.quantile_encode(X)
        return X

    def quantile_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X


# todo does not fit in schema since it is an ensemble of other encoders
class SummaryEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """Summary Encoding for categorical features.

    It's an encoder designed for creating richer representations by applying quantile encoding for a set of quantiles.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    quantiles: list
        list of floats indicating the statistical quantiles. Each element represent a column
    m: float
        this is the “m” in the m-probability estimate. Higher value of m results into stronger shrinking. M is non-negative. 0 for no smoothing.
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
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = SummaryEncoder(cols=["CentralAir", "Heating"], quantiles=[0.25, 0.5, 0.75]).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 11 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   float64
     1   MSSubClass     1460 non-null   float64
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   YearBuilt      1460 non-null   float64
     5   Heating_25     1460 non-null   float64
     6   Heating_50     1460 non-null   float64
     7   Heating_75     1460 non-null   float64
     8   CentralAir_25  1460 non-null   float64
     9   CentralAir_50  1460 non-null   float64
     10  CentralAir_75  1460 non-null   float64
    dtypes: float64(10), object(1)
    memory usage: 125.6+ KB
    None

    References
    ----------
    .. [1] Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, https://link.springer.com/chapter/10.1007%2F978-3-030-85529-1_14
    .. [2] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, equation 7, from https://dl.acm.org/citation.cfm?id=507538
    .. [3] On estimating probabilities in tree pruning, equation 1, from https://link.springer.com/chapter/10.1007/BFb0017010
    .. [4] Additive smoothing, from https://en.wikipedia.org/wiki/Additive_smoothing#Generalized_to_the_case_of_known_incidence_rates
    .. [5] Target encoding done the right way https://maxhalford.github.io/blog/target-encoding/
    """
    encoding_relation = util.EncodingRelation.ONE_TO_M

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_missing="value",
        handle_unknown="value",
        quantiles=(0.25, 0.75),
        m=1.0,
    ):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.use_default_cols = cols is None  # if True, even a repeated call of fit() will select string columns from X
        self.ordinal_encoder = None
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.feature_names_out_ = None
        self.quantiles = quantiles
        self.m = m
        self.encoder_list = None

    def fit(self, X, y):
        """Fits the encoder according to X and y by fitting the individual encoders.

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
        X, y = util.convert_inputs(X, y)

        if self.use_default_cols:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        rounded_percentiles = [round(quantile * 100) for quantile in self.quantiles]
        if len(rounded_percentiles) != len(set(rounded_percentiles)):
            raise ValueError("There are two quantiles that belong to the same rounded percentile")

        encoder_list = []
        for quantile in self.quantiles:
            enc = QuantileEncoder(
                verbose=self.verbose,
                cols=self.cols,
                drop_invariant=self.drop_invariant,
                return_df=True,  # always return df for individual encoders. If not desired this is handled below.
                handle_missing=self.handle_missing,
                handle_unknown=self.handle_unknown,
                quantile=quantile,
                m=self.m,
            )
            enc.fit(X.copy(), y)
            encoder_list.append(enc)
            self.drop_cols += enc.invariant_cols
        self.feature_names_out_ = reduce(
            operator.add,
            [
                [self._get_col_name(c, enc.quantile) for enc in encoder_list if c not in enc.invariant_cols]
                if c in self.cols
                else [c]
                for c in X.columns
            ],
        )
        self.encoder_list = encoder_list
        return self

    def transform(self, X, y=None, override_return_df=False):
        if self.encoder_list is None:
            raise ValueError("Must train encoder before it can be used to transform data.")
        X, y = util.convert_inputs(X, y)

        orig_cols = X.columns
        transformed_df = X.copy()
        for idx, encoder in enumerate(self.encoder_list):
            colname_mapping = {col: self._get_col_name(col, encoder.quantile) for col in self.cols}
            X_encoded = encoder.transform(X.copy()).rename(columns=colname_mapping)
            if idx == 0:
                transformed_df = X_encoded
            else:
                new_feat = X_encoded[[c for c in X_encoded.columns if c not in orig_cols]]
                transformed_df = pd.concat([transformed_df, new_feat], axis=1)
        feature_order = [c for c in self.get_feature_names_out() if c in transformed_df]
        transformed_df = transformed_df[feature_order]

        if self.return_df or override_return_df:
            return transformed_df
        else:
            return transformed_df.values

    def get_feature_names(self) -> List[str]:
        warnings.warn("`get_feature_names` is deprecated in all of sklearn. Use `get_feature_names_out` instead.",
                      category=FutureWarning)
        return self.get_feature_names_out()

    def get_feature_names_out(self):
        """
        Returns the names of all transformed / added columns.
        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not isinstance(self.feature_names_out_, list):
            raise NotFittedError("Must fit data first. Affected feature names are not known before.")
        else:
            return self.feature_names_out_

    def get_feature_names_in(self) -> List[str]:
        """
        Returns the names of all input columns present when fitting.
        These columns are necessary for the transform step.
       """
        if not isinstance(self.cols, list):
            raise NotFittedError("Estimator has to be fitted to return feature names.")
        else:
            return self.cols

    @staticmethod
    def _get_col_name(col: str, quantile: float) -> str:
        percentile = round(quantile * 100)
        return f"{col}_{percentile}"

"""A collection of shared utilities for all encoders, not intended for external use."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
import sklearn.base
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Tags

__author__ = 'willmcginnis'

X_type = np.ndarray | pd.DataFrame | list | np.generic | csr_matrix
y_type = list | pd.Series | np.ndarray | tuple | pd.DataFrame


def convert_cols_to_list(
    cols: pd.Series | np.ndarray | set | tuple | CategoricalDtype | str | int,
) -> list:
    """Convert columns to list.

    Parameters
    ----------
    cols: columns as Series, array, set, tuple, ...

    Returns
    -------
    columns as list.

    """
    if isinstance(cols, pd.Series):
        return cols.tolist()
    elif isinstance(cols, np.ndarray):
        return cols.tolist()
    elif np.isscalar(cols):
        return [cols]
    elif isinstance(cols, set):
        return list(cols)
    elif isinstance(cols, tuple):
        return list(cols)
    elif isinstance(cols, CategoricalDtype):
        return cols.astype(object).tolist()

    return cols


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Returns names of categorical columns in the DataFrame.

    These include columns of types: object, category, string, string[pyarrow].

    Parameters
    ----------
    df DataFrame

    Returns
    -------
    list of columns

    """
    obj_cols = []
    for col, dtype in df.dtypes.items():
        if is_object_dtype(dtype) or is_category(dtype) or is_string_dtype(dtype):
            # if not isinstance(col, str):
            #     raise ValueError(f'DataFrame column names must be strings not {col}.')
            obj_cols.append(col)

    if not obj_cols:
        msg = (
            'Warning: No categorical columns found. '
            "Calling 'transform' will only return input data."
        )
        print(msg)

    return obj_cols


def is_category(dtype: pd.core.dtypes.dtypes.ExtensionDtype) -> bool:
    """Check if dtype is pandas categorical type.

    Parameters
    ----------
    dtype pandas dtype

    Returns
    -------
    True if CategoricalDtype, False otherwise.

    """
    return isinstance(dtype, CategoricalDtype)


def convert_inputs(
    X: X_type,
    y: y_type | None,
    columns: Sequence = None,
    index: Sequence = None,
    deep: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Unite arraylike `X` and vectorlike `y` into a DataFrame and Series.

    If both are pandas types already, raises an error if their indexes do not match.
    If one is pandas, the returns will share that index.
    If neither is pandas, a default index will be used, unless `index` is passed.

    Parameters
    ----------
    X: arraylike
    y: listlike
    columns: listlike
        Specifies column names to use for `X`.
        Ignored if `X` is already a dataframe.
        If `None`, use the default pandas column names.
    index: listlike
        The index to use, if neither `X` nor `y` is a pandas type.
        (If one has an index, then this has no effect.)
        If `None`, use the default pandas index.
    deep: bool
        Whether to deep-copy `X`.
    """
    X_alt_index = y.index if isinstance(y, pd.Series) else index
    X = convert_input(X, columns=columns, deep=deep, index=X_alt_index)
    if y is not None:
        y = convert_input_vector(y, index=X.index)
        # N.B.: If either was already pandas, it keeps its index.

        if any(X.index != y.index):
            msg = (
                '`X` and `y` both have indexes, but they do not match. If you are shuffling '
                'your input data on purpose (e.g. via permutation_test_score) use '
                'np arrays instead of data frames / series'
            )
            raise ValueError(msg)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'The length of X is '
                + str(X.shape[0])
                + ' but length of y is '
                + str(y.shape[0])
                + '.'
            )
    return X, y


def convert_input(
    X: X_type, columns: Sequence = None, index: Sequence = None, deep: bool = False
) -> pd.DataFrame:
    """Unite data into a DataFrame.

    Objects that do not contain column names take the names from the argument.
    Optionally perform deep copy of the data.

    Parameters
    ----------
    X: data
    columns: column names to assign, ignored if data is already a data frame.
    index: index to use for the dataframe. Defaults to range(len(data)).
    deep: flag whether the data should be copied when creating the data frame.

    Returns
    -------
    A dataframe with the data and columns and index properly set.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, copy=deep)
        else:
            if columns is not None and np.size(X, 1) != len(columns):
                raise ValueError(
                    'The count of the column names does not correspond to the count of the columns'
                )
            if isinstance(X, list):
                X = pd.DataFrame(
                    X, columns=columns, copy=deep, index=index
                )  # lists are always copied, but for consistency, we still pass the argument
            elif isinstance(X, (np.generic, np.ndarray)):
                X = pd.DataFrame(X, columns=columns, copy=deep, index=index)
            elif isinstance(X, csr_matrix):
                X = pd.DataFrame(X.todense(), columns=columns, copy=deep, index=index)
            else:
                raise ValueError(f'Unexpected input type: {type(X)}')
    elif deep:
        X = X.copy(deep=True)

    return X


def convert_input_vector(y: y_type, index: Sequence) -> pd.Series:
    """Unite target data type into a Series.

    If the target is a Series or a DataFrame, we preserve its index.
    But if the target does not contain index attribute, we use the index from the argument.

    Parameters
    ----------
    y: target data to convert to series.
    index: index to be used for the series.

    Returns
    -------
    pd.Series containing the target.

    """
    if y is None:
        raise ValueError(
            'Supervised encoders need a target for the fitting. The target cannot be None'
        )
    if isinstance(y, pd.Series):
        return y
    elif isinstance(y, np.ndarray):
        if len(np.shape(y)) == 1:  # vector
            return pd.Series(y, name='target', index=index)
        elif len(np.shape(y)) == 2 and np.shape(y)[0] == 1:  # single row in a matrix
            return pd.Series(y[0, :], name='target', index=index)
        elif len(np.shape(y)) == 2 and np.shape(y)[1] == 1:  # single column in a matrix
            return pd.Series(y[:, 0], name='target', index=index)
        else:
            raise ValueError(f'Unexpected input shape: {np.shape(y)}')
    elif np.isscalar(y):
        raise ValueError('y must be a list, an np.ndarray or a pd.Series. Not a scalar')
    elif isinstance(y, list):
        if len(y) == 0:  # empty list
            return pd.Series(y, name='target', index=index, dtype=float)
        elif len(y) > 0 and not isinstance(y[0], list):  # vector
            return pd.Series(y, name='target', index=index)
        elif len(y) > 0 and isinstance(y[0], list) and len(y[0]) == 1:  # single row in a matrix

            def flatten(y):
                return [item for sublist in y for item in sublist]

            return pd.Series(flatten(y), name='target', index=index)
        elif (
            len(y) == 1 and len(y[0]) == 0 and isinstance(y[0], list)
        ):  # single empty column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=float)
        elif len(y) == 1 and isinstance(y[0], list):  # single column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=type(y[0][0]))
        else:
            raise ValueError('Unexpected input shape')
    elif isinstance(y, pd.DataFrame):
        if len(list(y)) == 0:  # empty DataFrame
            return pd.Series(name='target', index=index, dtype=float)
        if len(list(y)) == 1:  # a single column
            return y.iloc[:, 0]
        else:
            raise ValueError(f'Unexpected input shape: {y.shape}')
    else:
        return pd.Series(
            y, name='target', index=index
        )  # this covers tuples and other directly convertible types


def get_generated_cols(
    X_original: pd.DataFrame, X_transformed: pd.DataFrame, to_transform: list[Hashable]
) -> list[Hashable]:
    """
    Returns a list of the generated/transformed columns.

    Arguments:
        X_original: df
            the original (input) DataFrame.
        X_transformed: df
            the transformed (current) DataFrame.
        to_transform: [str]
            a list of columns that were transformed (as in the original DataFrame),
            commonly self.feature_names_in.

    Output:
        a list of columns that were transformed (as in the current DataFrame).
    """
    original_cols = list(X_original.columns)

    if len(to_transform) > 0:
        [original_cols.remove(c) for c in to_transform]

    current_cols = list(X_transformed.columns)
    if len(original_cols) > 0:
        [current_cols.remove(c) for c in original_cols]

    return current_cols


def flatten_reverse_dict(dict_to_flatten: dict) -> dict:
    """Flatten a dictionary into a tuple of nested keys.

    Parameters
    ----------
    dict_to_flatten

    Returns
    -------
    the flattened dictionary with tuples as keys indicating the hierarchy.

    """
    sep = '___'
    [flat_dict] = pd.json_normalize(dict_to_flatten, sep=sep).to_dict(orient='records')
    reversed_flat_dict = {v: tuple(k.split(sep)) for k, v in flat_dict.items()}
    return reversed_flat_dict


class EncodingRelation(Enum):
    """Relation of how many input features are encoded into how many output features."""

    # one input feature get encoded into one output feature
    ONE_TO_ONE = auto()
    # one input feature get encoded into as many output features as it has distinct values
    ONE_TO_N_UNIQUE = auto()
    # one input feature get encoded into m output features
    # that are not the number of distinct values
    ONE_TO_M = auto()
    # all N input features are encoded into M output features.
    # The encoding is done globally on all the input not on a per-feature basis
    N_TO_M = auto()


def get_docstring_output_shape(in_out_relation: EncodingRelation) -> str:
    """Find how many encoded features are expected.

    Parameters
    ----------
    in_out_relation

    Returns
    -------
    A string saying how many features to expect.

    """
    if in_out_relation == EncodingRelation.ONE_TO_ONE:
        return 'n_features'
    elif in_out_relation == EncodingRelation.ONE_TO_N_UNIQUE:
        return 'n_features * respective cardinality'
    elif in_out_relation == EncodingRelation.ONE_TO_M:
        return 'M features (n_features < M)'
    elif in_out_relation == EncodingRelation.N_TO_M:
        return 'M features (M can be anything)'


@dataclass
class EncoderTags(Tags):
    """Custom Tags for encoders."""

    predict_depends_on_y: bool = False

    @classmethod
    def from_sk_tags(cls, tags: Tags) -> EncoderTags:
        """Initialize EncoderTags from given sklearn Tags."""
        as_dict = {
            field.name: getattr(tags, field.name)
            for field in fields(tags)
        }
        return cls(**as_dict)

class BaseEncoder(BaseEstimator):
    """BaseEstimator class for all encoders.

    This follows the sklearn estimator / transformer pattern.
    """

    _dim: int | None
    cols: list[str]
    use_default_cols: bool
    handle_missing: str
    handle_unknown: str
    verbose: int
    drop_invariant: bool
    invariant_cols: list[str] = []
    return_df: bool
    supervised: bool
    encoding_relation: EncodingRelation

    INVARIANCE_THRESHOLD = (
        10e-5  # columns with variance less than this will be considered constant / invariant
    )

    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        **kwargs,
    ):
        """Initialize the encoder.

        Parameters
        ----------
        verbose: int
            integer indicating verbosity of output. 0 for none.
        cols: list
            a list of columns to encode, if None, all string and categorical columns
            will be encoded.
        drop_invariant: bool
            boolean for whether to drop columns with 0 variance.
        return_df: bool
            boolean for whether to return a pandas DataFrame from transform and inverse transform
            (otherwise it will be a numpy array).
        handle_missing: str
            how to handle missing values at fit time. Options are 'error', 'return_nan',
            and 'value'. Default 'value', which treat nans as a countable category at
            fit time.
        handle_unknown: str, int or dict of {column : option, ...}.
            how to handle unknown labels at transform time. Options are 'error'
            'return_nan', 'value' and int. Defaults to None which uses nan behaviour
            specified at fit time. Passing an int will fill with this int value.
        kwargs: dict.
            additional encoder specific parameters like regularisation.
        """
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.invariant_cols = []
        self.verbose = verbose
        # if True, even a repeated call of fit() will select string columns from X
        self.use_default_cols = cols is None
        # note that cols are only the columns to be encoded, feature_names_in_ are all columns
        self.cols = cols
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._dim = None

    def fit(self, X: X_type, y: y_type | None = None, **kwargs):
        """Fits the encoder according to X and y.

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
        X, y = convert_inputs(X, y)
        self._check_fit_inputs(X, y)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        if self.__sklearn_tags__().target_tags.required:
            if not is_numeric_dtype(y):
                self.lab_encoder_ = LabelEncoder()
                y = pd.Series(self.lab_encoder_.fit_transform(y), index=y.index)
            else:
                self.lab_encoder_ = None

        self._dim = X.shape[1]
        self._determine_fit_columns(X)

        if not set(self.cols).issubset(X.columns):
            raise ValueError('X does not contain the columns listed in cols')

        if self.handle_missing == 'error':
            if X[self.cols].isna().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self._fit(X, y, **kwargs)

        # for finding invariant columns transform without y (as is done on the test set)
        self.feature_names_out_ = None  # Issue#437
        X_transformed = self.transform(X, override_return_df=True)
        self.feature_names_out_ = X_transformed.columns.to_numpy()

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            generated_cols = get_generated_cols(X, X_transformed, self.cols)
            self.invariant_cols = [
                x for x in generated_cols if X_transformed[x].var() <= self.INVARIANCE_THRESHOLD
            ]
            self.feature_names_out_ = np.fromiter(
                (x for x in self.feature_names_out_ if x not in self.invariant_cols),
                dtype=self.feature_names_out_.dtype,
            )

        return self

    def _check_fit_inputs(self, X: X_type, y: y_type) -> None:
        if self.__sklearn_tags__().target_tags.required:
            if y is None:
                raise ValueError(
                    'Supervised encoders need a target for the fitting. The target cannot be None'
                )
            else:
                if y.isna().any():  # Target column should never have missing values
                    raise ValueError('The target column y must not contain missing values.')

    def _check_transform_inputs(self, df: pd.DataFrame) -> None:
        if self.handle_missing == 'error':
            if df[self.cols].isna().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise NotFittedError('Must train encoder before it can be used to transform data.')

        # then make sure that it is the right size
        if df.shape[1] != self._dim:
            raise ValueError(f'Unexpected input dimension {df.shape[1]}, expected {self._dim}')

    def _drop_invariants(
        self, df: pd.DataFrame, override_return_df: bool
    ) -> np.ndarray | pd.DataFrame:
        if self.drop_invariant:
            df = df.drop(columns=self.invariant_cols)

        if self.return_df or override_return_df:
            return df
        else:
            return df.to_numpy()

    def _determine_fit_columns(self, X: pd.DataFrame) -> None:
        """Determine columns used by encoder.

        Note that the implementation also deals with re-fitting the same encoder object
        with different columns.

        :param X: input data frame
        :return: none, sets self.cols as a side effect
        """
        # if columns aren't passed, just use every string column
        if self.use_default_cols:
            self.cols = get_categorical_cols(X)
        else:
            self.cols = convert_cols_to_list(self.cols)

    def get_feature_names(self) -> np.ndarray:
        """Deprecated method to get feature names. Use `get_feature_names_out` instead."""
        msg = (
            '`get_feature_names` is deprecated in all of sklearn. '
            'Use `get_feature_names_out` instead.'
        )
        warnings.warn(msg, category=FutureWarning, stacklevel=2)
        return self.get_feature_names_out()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get the names of all transformed / added columns.

        Note that in sklearn the get_feature_names_out function takes the feature_names_in
        as an argument and determines the output feature names using the input.
        A fit is usually not necessary and if so a NotFittedError is raised.
        We just require a fit all the time and return the fitted output columns.

        Returns
        -------
        feature_names: np.ndarray
            A numpy array with all feature names transformed or added.
            Note: potentially dropped features (because the feature is constant/invariant)
            are not included!

        """
        out_feats = getattr(self, 'feature_names_out_', None)
        if not isinstance(out_feats, np.ndarray):
            raise NotFittedError('Estimator has to be fitted to return feature names.')
        else:
            return out_feats

    def get_feature_names_in(self) -> np.ndarray:
        """Get the names of all input columns present when fitting.

        These columns are necessary for the transform step.
        """
        in_feats = getattr(self, 'feature_names_in_', None)
        if isinstance(in_feats, list):
            in_feats = np.array(in_feats)
        if not isinstance(in_feats, np.ndarray):
            raise NotFittedError('Estimator has to be fitted to return feature names.')
        else:
            return in_feats

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series | None, **kwargs): ...


class SupervisedTransformerMixin(sklearn.base.TransformerMixin):
    """Mixin for supervised transformers (with target)."""

    def __sklearn_tags__(self) -> EncoderTags:
        """Set scikit transformer tags."""
        sk_tags = super().__sklearn_tags__()
        tags = EncoderTags.from_sk_tags(sk_tags)
        tags.target_tags.required = True
        return tags

    def transform(self, X: X_type, y: y_type | None = None, override_return_df: bool = False):
        """Perform the transformation to new categorical data.

        Some encoders behave differently on whether y is given or not.
        This is mainly due to regularisation in order to avoid overfitting.
        On training data transform should be called with y, on test data without.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] or None
        override_return_df : bool
            override self.return_df to force to return a data frame

        Returns
        -------
        p : array or DataFrame, shape = [n_samples, n_features_out]
            Transformed values with encoding applied.

        """
        # first check the type
        X, y = convert_inputs(X, y, deep=True)
        self._check_transform_inputs(X)
        if y is not None and self.lab_encoder_ is not None:
            y = pd.Series(self.lab_encoder_.transform(y), index=y.index)

        if not list(self.cols):
            return X

        X = self._transform(X, y)

        return self._drop_invariants(X, override_return_df)

    @abstractmethod
    def _transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: ...

    def fit_transform(self, X: X_type, y: y_type | None = None, **fit_params):
        """Fit and transform using the target information.

        This also uses the target for transforming, not only for training.
        """
        if y is None:
            raise TypeError('fit_transform() missing argument: ' 'y' '')
        return self.fit(X, y, **fit_params).transform(X, y)


class UnsupervisedTransformerMixin(sklearn.base.TransformerMixin):
    """Mixin for Transformers without target information."""

    def transform(self, X: X_type, override_return_df: bool = False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        override_return_df : bool
            override self.return_df to force to return a data frame

        Returns
        -------
        p : array or DataFrame, shape = [n_samples, n_features_out]
            Transformed values with encoding applied.

        """
        # first check the type
        X = convert_input(X, deep=True)
        self._check_transform_inputs(X)

        if not list(self.cols):
            return X

        X = self._transform(X)
        return self._drop_invariants(X, override_return_df)

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

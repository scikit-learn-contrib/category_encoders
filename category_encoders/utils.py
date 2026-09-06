"""A collection of shared utilities for all encoders, not intended for external use."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import Callable, Hashable, Sequence

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

# Ordinal codes reserved for rows that cannot be mapped to a fitted category:
# UNKNOWN for labels never seen at fit time, MISSING for missing values when
# no missing category was seen at fit time.
UNKNOWN_SENTINEL = -1
MISSING_SENTINEL = -2

# Numeric types accepted for the handle_unknown/handle_missing strategies.
NUMERIC_SCALARS = (int, float, np.integer, np.floating)


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
        # composite member columns can be dropped from the output, so only remove
        # columns the transformed frame actually carries
        [current_cols.remove(c) for c in original_cols if c in current_cols]

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


def build_min_group_map(
    group_sizes: pd.Series,
    min_group_size: int | float,
    min_group_name: str | None,
    combine_min_nan_groups: bool | str | None,
) -> tuple[pd.Series, dict]:
    """Lump groups whose size is below `min_group_size` into one leftovers group.

    Pure helper behind CountEncoder's min_group_size lumping and the optional
    BaseEncoder-level min_group_size / min_group_name lumping.

    Parameters
    ----------
    group_sizes: pd.Series
        Size of each group, indexed by label. The index may contain NaN, which
        represents the group of missing values.
    min_group_size: int or float
        Resolved threshold a group must reach to be kept as its own group.
        Callers resolve int (absolute size) vs float (fraction of rows) semantics
        before calling.
    min_group_name: str or None
        Name of the leftovers group. When None, the names of the lumped labels are
        joined alphabetically with a `_` delimiter.
    combine_min_nan_groups: bool or 'force' or None
        Whether the missing-values group is folded into the leftovers group.
        True folds it in when it is itself below the threshold, 'force' always
        folds it in, False and None never do.

    Returns
    -------
    tuple[pd.Series, dict]
        The reduced group sizes, with the lumped groups replaced by the leftovers
        group, and the lumping map {label: lumped name}. The map is empty when no
        group was lumped. The input Series is not modified.
    """
    if combine_min_nan_groups is True:
        min_groups_idx = group_sizes < min_group_size
    elif combine_min_nan_groups == 'force':
        min_groups_idx = (group_sizes < min_group_size) | (group_sizes.index.isna())
    else:
        min_groups_idx = (group_sizes < min_group_size) & (~group_sizes.index.isna())

    min_groups_sum = group_sizes.loc[min_groups_idx].sum()

    if (
        min_groups_sum > 0
        and min_groups_idx.sum() > 1
        and not min_groups_idx.loc[~min_groups_idx.index.isna()].all()
    ):
        if isinstance(min_group_name, str):
            lumped_name = min_group_name
        else:
            lumped_name = '_'.join(
                [
                    str(idx)
                    for idx in group_sizes.loc[min_groups_idx].index.astype(str).sort_values()
                ]
            )
        lumping_map = dict.fromkeys(group_sizes.loc[min_groups_idx].index.tolist(), lumped_name)

        if not min_groups_idx.all():
            group_sizes = group_sizes.loc[~min_groups_idx]
            group_sizes[lumped_name] = min_groups_sum

        return group_sizes, lumping_map

    return group_sizes, {}


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

def evaluate_handle_callable(
    fn: Callable,
    value: float,
    mapping: pd.Series | pd.DataFrame,
    param_name: str,
) -> float:
    """Invoke a callable handle_unknown/handle_missing strategy and validate its result.

    The callable receives the row value (nan for the reserved sentinel rows) and
    the fitted mapping, and must return a numeric scalar so the finalized
    mapping stays numeric.
    """
    result = fn(value, mapping)
    if not isinstance(result, NUMERIC_SCALARS):
        raise ValueError(
            f'The callable passed for {param_name} must return a numeric scalar, '
            f'got {result!r} of type {type(result).__name__}.'
        )
    return result


def validate_scalar_handle_value(
    value: float,
    mapping: pd.Series | pd.DataFrame,
    param_name: str,
    col: str | None = None,
) -> float:
    """Reject a numeric handle_unknown/handle_missing value that collides with generated labels.

    A value equal to an already generated encoded value would make unknown or
    missing rows indistinguishable from that category at transform time.
    """
    if value in mapping.values:
        location = f' for column {col!r}' if col is not None else ''
        raise ValueError(
            f'{param_name}={value!r}{location} collides with an encoded category value; '
            'unknown or missing rows would become indistinguishable from that category. '
            'Choose a value outside the generated labels.'
        )
    return value


def finalize_encoding_mapping(
    estimate: pd.Series | pd.DataFrame,
    values: pd.Series,
    handle_unknown: str | float | Callable,
    handle_missing: str | float | Callable,
    prior: float,
) -> pd.Series | pd.DataFrame:
    """Finalize the unknown/missing rows of a per-column encoding mapping.

    Encoders that map ordinal codes to encoded values all share the same final
    step: the row for the unknown code (``UNKNOWN_SENTINEL``) and the row for
    the missing code (``MISSING_SENTINEL``) are filled according to the
    ``handle_unknown`` / ``handle_missing`` strategies. This helper implements
    that step once; ``estimate`` is modified in place and returned.

    Parameters
    ----------
    estimate: pd.Series or pd.DataFrame
        Per-code statistics computed by the encoder, indexed by ordinal code.
    values: pd.Series
        The fitted ordinal mapping (category -> ordinal code).
    handle_unknown: str, numeric scalar or callable
        Strategy for unseen categories: 'value' fills with ``prior``,
        'return_nan' fills with nan, a numeric scalar fills with that value,
        and a callable ``fn(value, mapping)`` fills with its result, evaluated
        once with ``value`` = nan and ``mapping`` = the fitted ordinal mapping.
    handle_missing: str, numeric scalar or callable
        Strategy for missing values: 'value' fills with ``prior``,
        'return_nan' fills with nan, a numeric scalar fills with that value,
        and a callable ``fn(value, mapping)`` fills with its result, evaluated
        once with ``value`` = nan and ``mapping`` = the fitted ordinal mapping.
    prior: float
        Encoder default (mean, quantile, zero, ...) used by the 'value' strategy.

    Returns
    -------
    The finalized mapping (same object as ``estimate``).

    """
    if handle_unknown == 'return_nan':
        estimate.loc[UNKNOWN_SENTINEL] = np.nan
    elif handle_unknown == 'value':
        estimate.loc[UNKNOWN_SENTINEL] = prior
    elif callable(handle_unknown):
        estimate.loc[UNKNOWN_SENTINEL] = evaluate_handle_callable(
            handle_unknown, np.nan, values, 'handle_unknown'
        )
    elif isinstance(handle_unknown, NUMERIC_SCALARS):
        validate_scalar_handle_value(handle_unknown, estimate, 'handle_unknown')
        estimate.loc[UNKNOWN_SENTINEL] = handle_unknown

    if handle_missing == 'return_nan':
        estimate.loc[values.loc[np.nan]] = np.nan
    elif handle_missing == 'value':
        estimate.loc[MISSING_SENTINEL] = prior
    elif callable(handle_missing):
        estimate.loc[MISSING_SENTINEL] = evaluate_handle_callable(
            handle_missing, np.nan, values, 'handle_missing'
        )
    elif isinstance(handle_missing, NUMERIC_SCALARS):
        validate_scalar_handle_value(handle_missing, estimate, 'handle_missing')
        estimate.loc[MISSING_SENTINEL] = handle_missing

    return estimate


class BaseEncoder(BaseEstimator):
    """BaseEstimator class for all encoders.

    This follows the sklearn estimator / transformer pattern.
    """

    _dim: int | None
    cols: list[str]
    use_default_cols: bool
    use_all_cols: bool
    handle_missing: str
    handle_unknown: str
    verbose: int
    drop_invariant: bool
    invariant_cols: list[str] = []
    return_df: bool
    supervised: bool
    encoding_relation: EncodingRelation
    min_group_size: int | float | None
    min_group_name: str | None
    combine_min_nan_groups: bool | str | None
    min_group_lumping_: dict[str, dict]
    composite_cols: list[tuple[str, ...]] | None
    keep_components: bool
    # fitted state: synthetic composite column name -> tuple of member columns
    composite_members_: dict[str, tuple[str, ...]] = {}

    INVARIANCE_THRESHOLD = (
        10e-5  # Deprecated: previously used as a variance threshold for invariant detection.
        # Now invariant columns are detected by checking nunique() <= 1, which is
        # scale-independent and handles normalized/proportion values correctly.
    )

    # Subclasses may override with a wider tuple, or set to None to opt out of
    # string validation (CountEncoder accepts dicts/ints; HashingEncoder uses a
    # sentinel string).
    _VALID_HANDLE_MISSING: tuple[str, ...] | None = ('error', 'return_nan', 'value')
    _VALID_HANDLE_UNKNOWN: tuple[str, ...] | None = ('error', 'return_nan', 'value')

    # Subclasses that implement their own min_group_size handling (CountEncoder)
    # set this to False to opt out of the base-level lumping hooks.
    _min_group_hooks_enabled: bool = True

    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
        composite_cols: list[tuple[str, ...]] | None = None,
        keep_components: bool = False,
        **kwargs,
    ):
        """Initialize the encoder.

        Parameters
        ----------
        verbose: int
            integer indicating verbosity of output. 0 for none.
        cols: list or "all"
            a list of columns to encode, if None, all string and categorical columns
            will be encoded. If "all", all columns will be encoded regardless of dtype.
        drop_invariant: bool
            boolean for whether to drop columns with 0 variance.
        return_df: bool
            boolean for whether to return a pandas DataFrame from transform and inverse transform
            (otherwise it will be a numpy array).
        handle_missing: str, int, float or callable
            how to handle missing values at fit time. Options are 'error', 'return_nan',
            and 'value'. Default 'value', which treat nans as a countable category at
            fit time. Passing a number uses it as the encoded value for missing values
            that were not seen at fit time. Passing a callable fn(value, mapping)
            computes the encoded value once per column when the mapping is finalized
            during fit, with `value` = np.nan and `mapping` the fitted
            category-to-ordinal-code mapping.
        handle_unknown: str, int, float, callable or dict of {column : option, ...}.
            how to handle unknown labels at transform time. Options are 'error',
            'return_nan', 'value' and int. Defaults to None which uses nan behaviour
            specified at fit time. Passing an int will fill with this int value.
            Passing a callable fn(value, mapping) computes the encoded value for
            unseen labels once per column when the mapping is finalized during fit
            (with `value` = np.nan). CountEncoder additionally accepts a dict of
            {column: option}.
        min_group_size: int or float, optional
            minimum group size needed for a category to be encoded as its own
            group; smaller categories are lumped into a single "leftovers" group
            before fitting. An int is an absolute group size, a float is a
            fraction of the number of rows. Default None, which disables lumping.
        min_group_name: str, optional
            name of the leftovers group created by `min_group_size`. Default
            None, in which case the names of the lumped categories are joined
            alphabetically with a `_` delimiter.
        combine_min_nan_groups: bool or 'force', optional
            whether the missing-values group is folded into the leftovers group
            created by `min_group_size`. True folds it in when it is itself below
            the threshold (the default), 'force' always folds it in, and False
            never does. 'force' requires `handle_missing` != 'return_nan'.
        composite_cols: list of tuples of str
            groups of columns to encode jointly, e.g. [('product', 'color')]. Each group
            becomes one synthetic column, named by joining the member names with '|',
            which is encoded like any other column (supervised encoders only). Rows
            where any member is missing get a missing composite value, so handle_missing
            applies as usual. The member columns themselves are dropped from the output
            unless keep_components is True. Default None (no composite columns).
        keep_components: bool
            if True, the member columns of every composite group are also encoded
            individually and kept in the output next to the composite columns.
            Default False.
        kwargs: dict.
            additional encoder specific parameters like regularisation.
        """
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.invariant_cols = []
        self.verbose = verbose
        # if True, even a repeated call of fit() will select string columns from X
        self.use_default_cols = cols is None
        # if True, even a repeated call of fit() will select all columns from X
        self.use_all_cols = isinstance(cols, str) and cols.lower() == 'all'
        # note that cols are only the columns to be encoded, feature_names_in_ are all columns
        self.cols = cols
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.min_group_size = min_group_size
        self.min_group_name = min_group_name
        self.combine_min_nan_groups = combine_min_nan_groups
        self.composite_cols = composite_cols
        self.keep_components = keep_components
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
        is_frame_input = isinstance(X, pd.DataFrame)
        X, y = convert_inputs(X, y)
        self._check_fit_inputs(X, y)
        self._validate_handle_strategies()
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        # dtypes of the fitted columns; re-attachment of arraylike input at transform
        # (GH #406) restores the object/category dtypes a plain array does not carry
        self.feature_dtypes_in_ = list(X.dtypes)

        if self.__sklearn_tags__().target_tags.required:
            if not is_numeric_dtype(y):
                self.lab_encoder_ = LabelEncoder()
                y = pd.Series(self.lab_encoder_.fit_transform(y), index=y.index)
            else:
                self.lab_encoder_ = None

        self._dim = X.shape[1]
        self._determine_fit_columns(X)
        X = self._apply_composite_cols(X)

        if not set(self.cols).issubset(X.columns):
            missing = [col for col in self.cols if col not in X.columns]
            msg = f'X does not contain the columns listed in cols: {missing}'
            if not is_frame_input:
                # A non-DataFrame fit input has no recoverable names, so the mismatch
                # cannot be fixed positionally (GH #406); point the user at the remedies.
                msg += (
                    ' The encoder received a non-DataFrame input at fit, where named columns'
                    ' cannot be recovered positionally. Either fit on a DataFrame containing'
                    ' these columns, make the upstream step emit a DataFrame (e.g. with'
                    " set_output(transform='pandas')), or pass the positional column indices"
                    ' (as integers) in cols.'
                )
            raise ValueError(msg)

        if self.handle_missing == 'error':
            if X[self.cols].isna().any().any():
                raise ValueError('Columns to be encoded cannot contain null')

        self.min_group_lumping_ = self._fit_min_group_lumping(X)
        if self.min_group_lumping_:
            # lump the training data itself so `_fit` sees the merged labels;
            # copy first because `X` may still be the caller's frame
            X = X.copy(deep=True)
            self._apply_min_group_lumping(X)

        self._fit(X, y, **kwargs)

        # for finding invariant columns transform without y (as is done on the test set)
        self.feature_names_out_ = None  # Issue#437
        # bypass set_output wrapping here; feature_names_out_ is not ready yet
        prev_output_config = getattr(self, '_sklearn_output_config', None)
        self._sklearn_output_config = {'transform': 'default'}
        try:
            X_transformed = self.transform(X, override_return_df=True)
        finally:
            if prev_output_config is None:
                del self._sklearn_output_config
            else:
                self._sklearn_output_config = prev_output_config
        self.feature_names_out_ = X_transformed.columns.to_numpy()

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            generated_cols = get_generated_cols(X, X_transformed, self.cols)
            self.invariant_cols = [
                x for x in generated_cols if X_transformed[x].nunique() <= 1
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

    def _validate_handle_strategies(self) -> None:
        """Raise ValueError if handle_missing/handle_unknown are unrecognised strings."""
        valid_missing = type(self)._VALID_HANDLE_MISSING
        if valid_missing is not None and isinstance(self.handle_missing, str):
            if self.handle_missing not in valid_missing:
                raise ValueError(
                    f'Unexpected handle_missing value {self.handle_missing!r} for '
                    f'{type(self).__name__}. Supported values: {sorted(valid_missing)}.'
                )
        valid_unknown = type(self)._VALID_HANDLE_UNKNOWN
        if valid_unknown is not None and isinstance(self.handle_unknown, str):
            if self.handle_unknown not in valid_unknown:
                raise ValueError(
                    f'Unexpected handle_unknown value {self.handle_unknown!r} for '
                    f'{type(self).__name__}. Supported values: {sorted(valid_unknown)}.'
                )

    def _check_transform_inputs(self, df: pd.DataFrame) -> None:
        # Fittedness is checked on the sklearn-standard n_features_in_ attribute, which
        # clone() never copies: a re-created encoder is unfitted by design, even when
        # constructor parameters leaked fitted state (Issue #232).
        if getattr(self, 'n_features_in_', None) is None:
            raise NotFittedError(
                f'This {type(self).__name__} instance is not fitted yet: call fit before '
                'using transform. If you passed a fitted encoder as a parameter of another '
                "estimator, scikit-learn's clone() re-created it unfitted. Fit the encoder "
                'inside the pipeline or cross-validation loop (for example as a Pipeline '
                'step), or implement __sklearn_clone__ on the wrapper estimator that holds '
                'it (scikit-learn >= 1.6). See the "Cloning and cross-validation" section '
                'of the documentation for details.'
            )

        missing = [col for col in self.cols if col not in df.columns]
        if missing:
            raise ValueError(f'X does not contain the columns listed in cols: {missing}')

        if self.handle_missing == 'error':
            if df[self.cols].isna().any().any():
                raise ValueError('Columns to be encoded cannot contain null')

        # then make sure that it is the right size
        # synthetic composite columns are appended before this check, so they widen
        # the frame beyond the fitted input width
        expected_dim = self._dim + len(self.composite_members_)
        if df.shape[1] != expected_dim:
            # DataFrames may carry extra pass-through columns beyond the fitted width:
            # encoders only touch self.cols and pass the rest through (GH #367). Narrower
            # frames keep the strict check, and arraylike input is already validated
            # positionally while the fitted names are re-attached.
            if df.shape[1] < expected_dim:
                raise ValueError(
                    f'Unexpected input dimension {df.shape[1]}, expected {expected_dim}'
                )

    def _transform_input_columns(self, X: X_type) -> list | None:
        """Resolve the column names to re-attach for a non-DataFrame transform input.

        Arraylike input carries no column names, so the names observed at fit are
        re-attached positionally (GH #406). Returns None when there is nothing to
        re-attach: DataFrame and Series input keep their own names, and an unfitted
        encoder fails downstream with the regular NotFittedError.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return None
        if not hasattr(self, 'feature_names_in_'):
            return None
        # ndarray and scipy sparse input expose .shape; flat lists and 1-d arrays
        # become a single column and keep their historical (nameless) behavior
        shape = getattr(X, 'shape', None)
        if shape is None or len(shape) != 2:
            return None
        if shape[1] != self.n_features_in_:
            # Arraylike input is positional, so its width check stays strict (GH #367).
            raise ValueError(
                f'Unexpected input dimension {shape[1]}, expected {self.n_features_in_}'
            )
        return self.feature_names_in_

    def _restore_fitted_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore the fitted dtypes on a positionally re-attached frame (GH #406).

        Only object and category columns carry encoding semantics that an ndarray
        loses when rebuilt into a DataFrame; numeric dtypes survive unchanged.
        """
        for position, dtype in enumerate(self.feature_dtypes_in_):
            column = df.columns[position]
            if is_object_dtype(dtype) or isinstance(dtype, CategoricalDtype):
                df[column] = df[column].astype(dtype)
        return df

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
        if self.use_all_cols:
            self.cols = X.columns.tolist()
        elif self.use_default_cols:
            self.cols = get_categorical_cols(X)
        else:
            self.cols = convert_cols_to_list(self.cols)

    def _validate_min_group_params(self) -> None:
        """Raise ValueError when the min_group_size lumping parameters conflict."""
        if self.min_group_name is not None and self.min_group_size is None:
            raise ValueError('`min_group_name` only works when `min_group_size` is set.')
        if self.combine_min_nan_groups is not None and self.combine_min_nan_groups not in [
            True,
            False,
            'force',
        ]:
            raise ValueError(
                "'combine_min_nan_groups' should be one of: ['force', True, False, None]."
            )
        if self.combine_min_nan_groups == 'force' and self.handle_missing == 'return_nan':
            raise ValueError(
                "Cannot have `handle_missing` == 'return_nan' and "
                "'combine_min_nan_groups' == 'force'."
            )

    def _fit_min_group_lumping(self, X: pd.DataFrame) -> dict[str, dict]:
        """Learn per-column lumping maps from min_group_size / min_group_name.

        Returns a mapping of column name to {original label: lumped name}; empty
        when lumping is disabled or no group falls below the threshold.
        """
        if not self._min_group_hooks_enabled:
            return {}
        self._validate_min_group_params()
        if self.min_group_size is None:
            return {}

        # None resolves to the library default: fold the missing group in when it
        # is itself below the threshold (matching CountEncoder's resolution).
        combine_nan = (
            self.combine_min_nan_groups if self.combine_min_nan_groups is not None else True
        )
        # int is an absolute group size, a float a fraction of the number of rows
        threshold = self.min_group_size
        if isinstance(threshold, float):
            threshold = threshold * len(X)

        lumping = {}
        for col in self.cols:
            # normalize None to NaN so both spellings of missing share one group
            group_sizes = X[col].fillna(np.nan).value_counts(dropna=False)
            _, lumping_map = build_min_group_map(
                group_sizes, threshold, self.min_group_name, combine_nan
            )
            if lumping_map:
                lumping[col] = lumping_map
        return lumping

    def _apply_min_group_lumping(self, X: pd.DataFrame) -> None:
        """Fold the labels of `X` into their lumped groups, in place."""
        for col, lumping_map in self.min_group_lumping_.items():
            # normalize None to NaN so both spellings of missing share one group
            X[col] = X[col].fillna(np.nan).map(lumping_map).fillna(X[col])
    def _apply_composite_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate composite_cols and append one synthetic '|'-joined column per group.

        The synthetic column enters self.cols and X, so the ordinary _fit machinery
        (inner ordinal encoding, per-column statistics) treats it like any other
        column, the way TargetEncoder.hierarchy already does for its HIER_ columns.
        Returns a new frame; the caller's X is never mutated.
        """
        self.composite_members_ = {}
        if not self.composite_cols:
            return X

        if not self.__sklearn_tags__().target_tags.required:
            raise ValueError(
                'composite_cols is only supported for supervised encoders, '
                f'not {type(self).__name__}.'
            )

        composites: dict[str, pd.Series] = {}
        for group in self._validate_composite_cols(X):
            name = '|'.join(group)
            composites[name] = self._join_composite(X, group, name)
            self.composite_members_[name] = tuple(group)

        X = pd.concat([X, pd.DataFrame(composites, index=X.index)], axis=1)
        for name in composites:
            if name not in self.cols:
                self.cols.append(name)
        if self.keep_components:
            for group in self.composite_members_.values():
                for member in group:
                    if member not in self.cols:
                        self.cols.append(member)
        return X

    def _validate_composite_cols(self, X: pd.DataFrame) -> list[tuple[str, ...]]:
        """Check the composite_cols parameter and return the groups as tuples."""
        if not isinstance(self.composite_cols, (list, tuple)):
            raise TypeError(
                'composite_cols must be a list of tuples of column names, '
                f"e.g. [('product', 'color')], got {self.composite_cols!r}."
            )
        groups: list[tuple[str, ...]] = []
        taken = set(X.columns)
        for group in self.composite_cols:
            if (
                not isinstance(group, tuple)
                or len(group) < 2
                or not all(isinstance(member, str) for member in group)
            ):
                raise ValueError(
                    'Each composite group must be a tuple of at least two column names, '
                    f"e.g. ('product', 'color'), got {group!r}."
                )
            name = '|'.join(group)
            missing = [member for member in group if member not in X.columns]
            if missing:
                raise ValueError(f'composite_cols references missing columns: {missing}')
            if name in taken:
                raise ValueError(
                    f'Composite column name {name!r} already exists in the input data.'
                )
            groups.append(group)
            taken.add(name)
        return groups

    @staticmethod
    def _join_composite(X: pd.DataFrame, members: tuple[str, ...], name: str) -> pd.Series:
        """Join the member columns of one composite group into a single column.

        Distinct member combinations must stay distinct after joining. Rows where any
        member is missing become missing so the regular handle_missing machinery
        applies to composites unchanged.
        """
        parts = X[list(members)].astype(str)
        joined = parts[members[0]]
        for member in members[1:]:
            joined = joined + '|' + parts[member]
        if len(set(joined)) < len(set(map(tuple, parts.to_numpy()))):
            raise ValueError(
                f'Composite column {name!r} is ambiguous: the "|" separator occurs in '
                'the data, so distinct value combinations collide after joining.'
            )
        joined[X[list(members)].isna().any(axis=1)] = np.nan
        return joined

    def _add_composite_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rebuild the synthetic composite columns on the transform input.

        Synthetic columns are always derived from the member columns, so transform
        cannot drift from fit; a frame that already carries them (the fit-time frame)
        gets them rebuilt in place at the end of the column list.
        """
        if not self.composite_members_:
            return X
        members = [m for group in self.composite_members_.values() for m in group]
        missing_inputs = [m for m in members if m not in X.columns]
        if missing_inputs:
            raise ValueError(
                'composite_cols member columns are missing from the transform input: '
                f'{missing_inputs}'
            )
        composites = {
            name: self._join_composite(X, group, name)
            for name, group in self.composite_members_.items()
        }
        X = X.drop(columns=[name for name in composites if name in X.columns])
        return pd.concat([X, pd.DataFrame(composites, index=X.index)], axis=1)

    def _drop_composite_members(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop composite member columns from the output unless keep_components."""
        if self.keep_components or not self.composite_members_:
            return df
        members = [m for group in self.composite_members_.values() for m in group]
        drop = [member for member in members if member in df.columns]
        return df.drop(columns=drop) if drop else df

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

        Some encoders behave differently on whether or not y is given.
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

        Notes
        -----
        If the encoder was fitted on a DataFrame, arraylike input (e.g. the numpy
        array emitted by the previous step of a scikit-learn pipeline) is accepted:
        the fitted column names are re-attached positionally, so the result matches
        transforming the equivalent DataFrame (GH #406). A DataFrame may additionally
        carry extra pass-through columns beyond the encoded ones (GH #367).

        """
        # first check the type
        columns = self._transform_input_columns(X)
        X, y = convert_inputs(X, y, columns=columns, deep=True)
        if columns is not None:
            X = self._restore_fitted_dtypes(X)
        X = self._add_composite_columns(X)
        self._check_transform_inputs(X)
        if y is not None and self.lab_encoder_ is not None:
            y = pd.Series(self.lab_encoder_.transform(y), index=y.index)

        if not list(self.cols):
            return X if (self.return_df or override_return_df) else X.to_numpy()

        self._apply_min_group_lumping(X)
        X = self._transform(X, y)
        X = self._drop_composite_members(X)

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

        Notes
        -----
        If the encoder was fitted on a DataFrame, arraylike input (e.g. the numpy
        array emitted by the previous step of a scikit-learn pipeline) is accepted:
        the fitted column names are re-attached positionally, so the result matches
        transforming the equivalent DataFrame (GH #406). A DataFrame may additionally
        carry extra pass-through columns beyond the encoded ones (GH #367).

        """
        # first check the type
        columns = self._transform_input_columns(X)
        X = convert_input(X, columns=columns, deep=True)
        if columns is not None:
            X = self._restore_fitted_dtypes(X)
        self._check_transform_inputs(X)

        if not list(self.cols):
            return X if (self.return_df or override_return_df) else X.to_numpy()

        self._apply_min_group_lumping(X)
        X = self._transform(X)
        return self._drop_invariants(X, override_return_df)

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

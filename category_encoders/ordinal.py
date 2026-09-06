"""Ordinal or label encoding."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

import category_encoders.utils as util

__author__ = 'willmcginnis'


class OrdinalEncoder( util.UnsupervisedTransformerMixin,util.BaseEncoder):
    """Encodes categorical features as ordinal, in one ordered feature.

    Ordinal encoding uses a single column of integers to represent the classes.
    An optional mapping dict can be passed in; in this case, we use the knowledge that there is
    some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    mapping: list of dicts
        a mapping of class to label to use for the encoding, optional.
        the dict contains the keys 'col' and 'mapping'.
        the value of 'col' should be the feature name.
        the value of 'mapping' should be a dictionary or pd.Series of 'original_label' to
        'encoded_label'.
        example mapping: [
            {'col': 'col1', 'mapping': {None: 0, 'a': 1, 'b': 2}},
            {'col': 'col2', 'mapping': {None: 0, 'x': 1, 'y': 2}}
        ]
    handle_unknown: str, int, float or callable
        options are 'error', 'return_nan' and 'value', defaults to 'value',
        which will impute the category -1. A number is imputed directly, and
        a callable fn(value, mapping) is evaluated per unseen value at transform
        time, where `value` is the unseen label and `mapping` is the fitted
        category-to-label mapping.
    handle_missing: str, int, float or callable
        options are 'error', 'return_nan', and 'value, default to 'value',
        which treat nan as a category at fit time,
        or -2 at transform time if nan is not a category during fit.
        A number replaces the -2 default for missing values that were not seen
        at fit time, and a callable fn(value, mapping) is evaluated once per
        column with `value` = np.nan and `mapping` the fitted
        category-to-label mapping.
    index_start: int
        integer at which to start labelling the categories. Defaults to 1.
        Set to 0 for zero-indexed labels, which can be convenient when feeding
        the encoded values into models that expect zero-indexed inputs such as
        embedding layers.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = OrdinalEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
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
     5   Heating      1460 non-null   int64
     6   CentralAir   1460 non-null   int64
    dtypes: float64(4), int64(2), object(1)
    memory usage: 80.0+ KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://ibgwww.colorado.edu/~carey/p5741ndir/Coding_Categorical_Variables.pdf
    """

    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(
        self,
        verbose: int = 0,
        mapping: list[dict[str, str | dict | pd.Series]] | None = None,
        cols: list[str] = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        index_start: int = 1,
        min_group_size: int | float | None = None,
        min_group_name: str | None = None,
        combine_min_nan_groups: bool | str | None = None,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            min_group_size=min_group_size,
            min_group_name=min_group_name,
            combine_min_nan_groups=combine_min_nan_groups,
        )
        self.mapping_supplied = mapping is not None
        if self.mapping_supplied:
            mapping = self._validate_supplied_mapping(mapping)
        self.mapping = mapping
        self.index_start = index_start

    @property
    def category_mapping(self) -> list[dict[str, str | dict | pd.Series]] | None:
        """The underlying category mapping."""
        return self.mapping

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs) -> None:
        # reset mapping in case of refit
        if self.mapping_supplied:
            return
        self.mapping = None
        _, categories = self.ordinal_encoding(
            X,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing,
            index_start=self.index_start,
        )
        self.mapping = categories

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X, _ = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing,
            index_start=self.index_start,
        )
        return X

    def inverse_transform(self, X_in: util.X_type) -> pd.DataFrame | np.ndarray:
        """Perform the inverse transformation to encoded data.

        Will attempt best case reconstruction, which means it will return nan for handle_missing
        and handle_unknown settings that break the bijection.
        We issue warnings when some of those cases occur.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """
        # fail fast
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        # first check the type and make deep copy; re-attach the fitted output names
        # for arraylike input, matching the BaseN/OneHot inverse_transform precedent
        X = util.convert_input(X_in, columns=self.feature_names_out_, deep=True)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError(
                    f'Unexpected input dimension {X.shape[1]}, the attribute drop_invariant should '
                    'be False when transforming the data'
                )
            else:
                raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')

        if not list(self.cols):
            return X if self.return_df else X.to_numpy()

        if self.handle_unknown == 'value':
            for col in self.cols:
                if any(X[col] == util.UNKNOWN_SENTINEL):
                    warnings.warn(
                        'inverse_transform is not supported because transform impute '
                        f'the unknown category {util.UNKNOWN_SENTINEL} when encode {col}',
                        stacklevel=4,
                    )

        if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
            for col in self.cols:
                if X[col].isna().any():
                    warnings.warn(
                        'inverse_transform is not supported because transform impute '
                        f'the unknown category nan when encode {col}',
                        stacklevel=4,
                    )

        for switch in self.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X if self.return_df else X.to_numpy()

    @staticmethod
    def ordinal_encoding(
        X_in: pd.DataFrame,
        mapping: list[dict[str, str | dict | pd.Series]] | None = None,
        cols: list[str] = None,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        index_start: int = 1,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Ordinal encoding uses a single column of integers to represent the classes.

        An optional mapping dict can be passed in, in this case we use the knowledge that there
        is some true order to the classes themselves.
        Otherwise, the classes are assumed to have no true order and integers are selected
        at random.
        """
        X = X_in

        if cols is None:
            cols = X.columns

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                column = switch.get('col')
                col_mapping = switch['mapping']
                raw_values = X[column] if callable(handle_unknown) else None
                X[column] = OrdinalEncoder._map_column(X[column], col_mapping)
                X[column] = OrdinalEncoder._apply_unknown_policy(
                    X[column], column, handle_unknown, raw_values, col_mapping
                )
                X[column] = OrdinalEncoder._apply_missing_policy(
                    X[column], handle_missing, column, col_mapping
                )
        else:
            mapping_out = []
            for col in cols:
                mapping_out.append(
                    {
                        'col': col,
                        'mapping': OrdinalEncoder._fit_column_mapping(
                            X[col], handle_missing, index_start
                        ),
                        'data_type': X[col].dtype,
                    }
                )

        return X, mapping_out

    @staticmethod
    def _map_column(values: pd.Series, col_mapping: pd.Series) -> pd.Series:
        """Map one column through its fitted category-to-code mapping."""
        # Convert to object to accept np.nan (dtype string doesn't)
        # fillna changes None and pd.NA to np.nan
        try:
            with pd.option_context('future.no_silent_downcasting', True):
                values = values.astype('object').fillna(np.nan).map(col_mapping)
        except pd._config.config.OptionError:  # old pandas versions
            values = values.astype('object').fillna(np.nan).map(col_mapping)
        if util.is_category(values.dtype):
            nan_identity = col_mapping.loc[col_mapping.index.isna()].array[0]
            values = values.cat.add_categories(nan_identity)
            values = values.fillna(nan_identity)
        try:
            values = values.astype(int)
        except ValueError:
            values = values.astype(float)
        return values

    @staticmethod
    def _apply_unknown_policy(
        values: pd.Series,
        column: str,
        handle_unknown: str | float | Callable,
        raw_values: pd.Series | None,
        col_mapping: pd.Series,
    ) -> pd.Series:
        """Resolve unseen categories after mapping: impute, raise, or defer to a scalar/callable."""
        unknown_mask = values.isna()
        if handle_unknown == 'value':
            return values.fillna(util.UNKNOWN_SENTINEL)
        if handle_unknown == 'error':
            if unknown_mask.any():
                raise ValueError(f'Unexpected categories found in column {column}')
        elif callable(handle_unknown):
            # each unseen raw value is passed to the callable individually, so unlike
            # handle_missing (a single conceptual nan) this bypasses evaluate_handle_callable
            filled = raw_values[unknown_mask].map(lambda value: handle_unknown(value, col_mapping))
            values = values.mask(unknown_mask, filled)
        elif isinstance(handle_unknown, util.NUMERIC_SCALARS):
            util.validate_scalar_handle_value(handle_unknown, col_mapping, 'handle_unknown', column)
            values = values.fillna(handle_unknown)
        return values

    @staticmethod
    def _apply_missing_policy(
        values: pd.Series,
        handle_missing: str | float | Callable,
        column: str,
        col_mapping: pd.Series,
    ) -> pd.Series:
        """Map the missing sentinel to NaN, or to a scalar/callable result, per handle_missing."""
        if handle_missing == 'return_nan':
            return_nan_series = pd.Series(data=[np.nan], index=[util.MISSING_SENTINEL])
            return values.map(return_nan_series).where(values == util.MISSING_SENTINEL, values)
        if callable(handle_missing):
            missing_value = util.evaluate_handle_callable(
                handle_missing, np.nan, col_mapping, 'handle_missing'
            )
            return values.mask(values == util.MISSING_SENTINEL, missing_value)
        if isinstance(handle_missing, util.NUMERIC_SCALARS):
            util.validate_scalar_handle_value(handle_missing, col_mapping, 'handle_missing', column)
            return values.mask(values == util.MISSING_SENTINEL, handle_missing)
        return values

    @staticmethod
    def _get_categories(values: pd.Series) -> list:
        """Collect the unique categories of one column, NaN last."""
        nan_identity = np.nan
        categories = values.unique()
        # make nan last category
        if pd.isna(categories).any():
            categories = [c for c in categories if not pd.isna(c)] + [nan_identity]
        else:
            categories = list(categories)
        if util.is_category(values.dtype):
            # Avoid using pandas category dtype meta-data if possible, see #235, #238.
            if values.dtype.ordered:
                category_set = set(
                    categories
                )  # convert to set for faster membership checks c.f. #407
                categories = [c for c in values.dtype.categories if c in category_set]
            if values.isna().any():
                categories += [np.nan]
        return categories

    @staticmethod
    def _fit_column_mapping(
        values: pd.Series, handle_missing: str | float | Callable, index_start: int
    ) -> pd.Series:
        """Build the category-to-code mapping of one column from the fit data."""
        nan_identity = np.nan
        categories = OrdinalEncoder._get_categories(values)

        index = pd.Series(categories).fillna(nan_identity).unique()

        data = pd.Series(
            index=index,
            data=range(index_start, len(index) + index_start),
        )

        if handle_missing == 'value' and ~data.index.isna().any():
            data.loc[nan_identity] = util.MISSING_SENTINEL
        elif handle_missing == 'return_nan':
            data.loc[nan_identity] = util.MISSING_SENTINEL
        elif callable(handle_missing) or isinstance(handle_missing, util.NUMERIC_SCALARS):
            # reserve the missing code so transform can replace it with
            # the user's scalar / callable result
            data.loc[nan_identity] = util.MISSING_SENTINEL

        return data

    def _validate_supplied_mapping(
        self, supplied_mapping: list[dict[str, str | dict | pd.Series]]
    ) -> list[dict[str, str | pd.Series]]:
        """
        Validate the supplied mapping and convert the actual mapping per column to a pandas series.

        :param supplied_mapping: mapping as list of dicts.
             They actual mapping can be either a dict or pd.Series
        :return: the mapping with all actual mappings being pandas series.
        """
        msg = (
            'Invalid supplied mapping, must be of type List[Dict[str, Union[Dict, pd.Series]]].'
            'For an example refer to the documentation'
        )
        if not isinstance(supplied_mapping, list):
            raise ValueError(msg)
        for mapping_el in supplied_mapping:
            if not isinstance(mapping_el, dict):
                raise ValueError(msg)
            if 'col' not in mapping_el:
                raise KeyError("Mapping must contain a key 'col' for each column to encode")
            if 'mapping' not in mapping_el:
                raise KeyError("Mapping must contain a key 'mapping' for each column to encode")
            mapping = mapping_el['mapping']
            if isinstance(mapping_el, dict):
                # convert to dict in order to standardise
                mapping_el['mapping'] = pd.Series(mapping)
            if 'data_type' not in mapping_el:
                mapping_el['data_type'] = mapping_el['mapping'].index.dtype
        return supplied_mapping

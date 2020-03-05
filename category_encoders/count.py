"""Count Encoder"""
from __future__ import division

import numpy as np
import pandas as pd
import category_encoders.utils as util

from copy import copy
from sklearn.base import BaseEstimator, TransformerMixin


__author__ = 'joshua t. dunn'


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, cols=None, drop_invariant=False,
                 return_df=True, handle_unknown=None,
                 handle_missing='count',
                 min_group_size=None, combine_min_nan_groups=None,
                 min_group_name=None, normalize=False):
        """Count encoding for categorical features.

        For a given categorical feature, replace the names of the groups
        with the group counts.

        Parameters
        ----------

        verbose: int
            integer indicating verbosity of output. 0 for none.
        cols: list
            a list of columns to encode, if None, all string and categorical columns
            will be encoded.
        drop_invariant: bool
            boolean for whether or not to drop columns with 0 variance.
        return_df: bool
            boolean for whether to return a pandas DataFrame from transform
            (otherwise it will be a numpy array).
        handle_missing: str
            how to handle missing values at fit time. Options are 'error', 'return_nan',
            and 'count'. Default 'count', which treat NaNs as a countable category at
            fit time.
        handle_unknown: str, int or dict of.
            how to handle unknown labels at transform time. Options are 'error'
            'return_nan' and an int. Defaults to None which uses NaN behaviour
            specified at fit time. Passing an int will fill with this int value.
        normalize: bool or dict of.
            whether to normalize the counts to the range (0, 1). See Pandas `value_counts`
            for more details.
        min_group_size: int, float or dict of.
            the minimal count threshold of a group needed to ensure it is not
            combined into a "leftovers" group. If float in the range (0, 1),
            `min_group_size` is calculated as int(X.shape[0] * min_group_size).
            Note: This value may change type based on the `normalize` variable. If True
            this will become a float. If False, it will be an int.
        min_group_name: None, str or dict of.
            Set the name of the combined minimum groups when the defaults become
            too long. Default None. In this case the category names will be joined
            alphabetically with a `_` delimiter.
            Note: The default name can be long ae may keep changing, for example, 
            in cross-validation.
        combine_min_nan_groups: bool or dict of.
            whether to combine the leftovers group with NaN group. Default True. Can
            also be forced to combine with 'force' meaning small groups are effectively
            counted as NaNs. Force can only used when 'handle_missing' is 'count' or 'error'.
            Note: Will not force if it creates an binary or invariant column.


        Example
        -------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_boston
        >>> from category_encoders import CountEncoder

        >>> bunch = load_boston()
        >>> y = bunch.target
        >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        >>> enc = CountEncoder(cols=['CHAS', 'RAD']).fit(X, y)
        >>> numeric_dataset = enc.transform(X)

        >>> print(numeric_dataset.info())
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 506 entries, 0 to 505
        Data columns (total 13 columns):
        CRIM       506 non-null float64
        ZN         506 non-null float64
        INDUS      506 non-null float64
        CHAS       506 non-null int64
        NOX        506 non-null float64
        RM         506 non-null float64
        AGE        506 non-null float64
        DIS        506 non-null float64
        RAD        506 non-null int64
        TAX        506 non-null float64
        PTRATIO    506 non-null float64
        B          506 non-null float64
        LSTAT      506 non-null float64
        dtypes: float64(11), int64(2)
        memory usage: 51.5 KB
        None

        References
        ----------

        """
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.normalize = normalize
        self.min_group_size = min_group_size
        self.min_group_name = min_group_name
        self.combine_min_nan_groups = combine_min_nan_groups

        self._check_set_create_attrs()

        self._min_group_categories = {}
        self._normalize = {}
        self._min_group_name = {}
        self._combine_min_nan_groups = {}
        self._min_group_size = {}
        self._handle_unknown = {}
        self._handle_missing = {}

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X.

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

        # first check the type
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        self._check_set_create_dict_attrs()

        self._fit_count_encode(X, y)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [
                x for x in generated_cols if X_temp[x].var() <= 10e-5
            ]

        return self

    def transform(self, X, y=None):
        """Perform the transformation to new categorical data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
            
        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.
        """

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.'
            )

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError(
                'Unexpected input dimension %d, expected %d'
                % (X.shape[1], self._dim,)
            )

        if not list(self.cols):
            return X

        X, _ = self._transform_count_encode(X, y)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def _fit_count_encode(self, X_in, y):
        """Perform the count encoding."""
        X = X_in.copy(deep=True)

        if self.cols is None:
            self.cols = X.columns.values

        self.mapping = {}

        for col in self.cols:
            if X[col].isnull().any():
                if self._handle_missing[col] == 'error':
                    raise ValueError(
                        'Missing data found in column %s at fit time.'
                        % (col,)
                    )

                elif self._handle_missing[col] not in ['count', 'return_nan',  'error', None]:
                    raise ValueError(
                        '%s key in `handle_missing` should be one of: '
                        ' `count`, `return_nan` and `error` not `%s`.'
                        % (col, str(self._handle_missing[col]))
                    )

            self.mapping[col] = X[col].value_counts(
                normalize=self._normalize[col],
                dropna=False
            )

            self.mapping[col].index = self.mapping[col].index.astype(object)

            if self._handle_missing[col] == 'return_nan':
                self.mapping[col][np.NaN] = np.NaN

        if any([val is not None for val in self._min_group_size.values()]):
            self.combine_min_categories(X)

    def _transform_count_encode(self, X_in, y):
        """Perform the transform count encoding."""
        X = X_in.copy(deep=True)
        X.loc[:, self.cols] = X.fillna(value=np.nan)

        for col in self.cols:
            if self._min_group_size is not None:
                if col in self._min_group_categories.keys():
                    X[col] = (
                        X[col].map(self._min_group_categories[col])
                        .fillna(X[col])
                    )

            X[col] = X[col].map(self.mapping[col])

            if isinstance(self._handle_unknown[col], (int, np.integer)):
                X[col] = X[col].fillna(self._handle_unknown[col])
            elif (
                self._handle_unknown[col] == 'error'
                and X[col].isnull().any()
            ):

                raise ValueError(
                    'Missing data found in column %s at transform time.'
                    % (col,)
                )

        return X, self.mapping

    def combine_min_categories(self, X):
        """Combine small categories into a single category."""
        for col, mapper in self.mapping.items():

            if self._normalize[col] and isinstance(self._min_group_size[col], int):
                self._min_group_size[col] = self._min_group_size[col] / X.shape[0]
            elif not self._normalize and isinstance(self._min_group_size[col], float):
                self._min_group_size[col] = self._min_group_size[col] * X.shape[0]

            if self._combine_min_nan_groups[col] is True:
                min_groups_idx = mapper < self._min_group_size[col]
            elif self._combine_min_nan_groups[col] == 'force':
                min_groups_idx = (
                    (mapper < self._min_group_size[col])
                    | (mapper.index.isnull())
                )
            else:
                min_groups_idx = (
                    (mapper < self._min_group_size[col])
                    & (~mapper.index.isnull())
                )

            min_groups_sum = mapper.loc[min_groups_idx].sum()

            if (
                min_groups_sum > 0
                and min_groups_idx.sum() > 1
                and not min_groups_idx.loc[~min_groups_idx.index.isnull()].all()
            ):
                if isinstance(self._min_group_name[col], str):
                    min_group_mapper_name = self._min_group_name[col]
                else:
                    min_group_mapper_name = '_'.join([
                        str(idx)
                        for idx
                        in mapper.loc[min_groups_idx].index.astype(str).sort_values()
                    ])

                self._min_group_categories[col] = {
                    cat: min_group_mapper_name
                    for cat
                    in mapper.loc[min_groups_idx].index.tolist()
                }

                if not min_groups_idx.all():
                    mapper = mapper.loc[~min_groups_idx]
                    mapper[min_group_mapper_name] = min_groups_sum

                self.mapping[col] = mapper

    def _check_set_create_attrs(self):
        """Check attributes setting that don't play nicely `self.cols`."""
        if not (
            (self.combine_min_nan_groups in ['force', True, False, None])
            or isinstance(self.combine_min_nan_groups, dict)
        ):
            raise ValueError(
                "'combine_min_nan_groups' should be one of: "
                "['force', True, False, None] or type dict."
            )

        if (
            self.handle_missing == 'return_nan'
            and self.combine_min_nan_groups == 'force'
        ):
            raise ValueError(
                "Cannot have `handle_missing` == 'return_nan' and "
                "'combine_min_nan_groups' == 'force' for all columns."
            )
        
        if (
            self.combine_min_nan_groups is not None
            and self.min_group_size is None
        ):
            raise ValueError(
                "`combine_min_nan_groups` only works when `min_group_size` "
                "is set for all columns."
            )

        if (
            self.min_group_name is not None
            and self.min_group_size is None
        ):
            raise ValueError(
                "`min_group_name` only works when `min_group_size` is set "
                "for all columns."
            )

        if self.combine_min_nan_groups is None:
            self.combine_min_nan_groups = True

    def _check_set_create_dict_attrs(self):
        """Check attributes that can be dicts and format for all `self.cols`."""
        dict_attrs = {
            'normalize': False,
            'min_group_name': None,
            'combine_min_nan_groups': True,
            'min_group_size': None,
            'handle_unknown': 'count',
            'handle_missing': None,
        }

        for attr_name, attr_default in dict_attrs.items():
            attr = copy(getattr(self, attr_name))
            if isinstance(attr, dict):
                for col in self.cols:
                    if col not in attr:
                        attr[col] = attr_default
                setattr(self, '_' + attr_name, attr)
            else:
                attr_dict = {}
                for col in self.cols:
                    attr_dict[col] = attr
                setattr(self, '_' + attr_name, attr_dict)

        for col in self.cols:
            if (
                self._handle_missing[col] == 'return_nan'
                and self._combine_min_nan_groups[col] == 'force'
            ):
                raise ValueError(
                    "Cannot have `handle_missing` == 'return_nan' and "
                    "'combine_min_nan_groups' == 'force' for columns `%s`."
                    % (col,)
                )
            
            if (
                self._combine_min_nan_groups[col] is not True
                and self._min_group_size[col] is None
            ):
                raise ValueError(
                    "`combine_min_nan_groups` only works when `min_group_size`"
                    "is set for column %s."
                    % (col,)
                )

            if (
                self._min_group_name[col] is not None
                and self._min_group_size[col] is None
            ):
                raise ValueError(
                    "`min_group_name` only works when `min_group_size`"
                    "is set for column %s."
                    % (col,)
                )

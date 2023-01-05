"""Count Encoder"""
import numpy as np
import pandas as pd
import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

from copy import copy


__author__ = 'joshua t. dunn'


class CountEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False,
                 return_df=True, handle_unknown='value',
                 handle_missing='value',
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
            and 'value'. Default 'value', which treat NaNs as a countable category at
            fit time.
        handle_unknown: str, int or dict of {column : option, ...}.
            how to handle unknown labels at transform time. Options are 'error'
            'return_nan', 'value' and int. Defaults to None which uses NaN behaviour
            specified at fit time. Passing an int will fill with this int value.
        normalize: bool or dict of {column : bool, ...}.
            whether to normalize the counts to the range (0, 1). See Pandas `value_counts`
            for more details.
        min_group_size: int, float or dict of {column : option, ...}.
            the minimal count threshold of a group needed to ensure it is not
            combined into a "leftovers" group. Default value is 0.01. 
            If float in the range (0, 1),
            `min_group_size` is calculated as int(X.shape[0] * min_group_size).
            Note: This value may change type based on the `normalize` variable. If True
            this will become a float. If False, it will be an int.
        min_group_name: None, str or dict of {column : option, ...}.
            Set the name of the combined minimum groups when the defaults become
            too long. Default None. In this case the category names will be joined
            alphabetically with a `_` delimiter.
            Note: The default name can be long and may keep changing, for example, 
            in cross-validation.
        combine_min_nan_groups: bool or dict of {column : bool, ...}.
            whether to combine the leftovers group with NaN group. Default True. Can
            also be forced to combine with 'force' meaning small groups are effectively
            counted as NaNs. Force can only be used when 'handle_missing' is 'value' or 'error'.
            Note: Will not force if it creates a binary or invariant column.


        Example
        -------
        >>> import pandas as pd
        >>> from sklearn.datasets import fetch_openml
        >>> from category_encoders import CountEncoder

        >>> bunch = fetch_openml(name="house_prices", as_frame=True)
        >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
        >>> y = bunch.target
        >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
        >>> enc = CountEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
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

        """
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.mapping = None
        self.normalize = normalize
        self.min_group_size = min_group_size
        self.min_group_name = min_group_name
        self.combine_min_nan_groups = combine_min_nan_groups
        self.ordinal_encoder = None

        self._check_set_create_attrs()

        self._min_group_categories = {}
        self._normalize = {}
        self._min_group_name = {}
        self._combine_min_nan_groups = {}
        self._min_group_size = {}
        self._handle_unknown = {}
        self._handle_missing = {}

    def _fit(self, X, y=None, **kwargs):
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        self._check_set_create_dict_attrs()
        self._fit_count_encode(X_ordinal, y)
        return self

    def _transform(self, X):
        for col in self.cols:
            # Treat None as np.nan
            X[col] = pd.Series([el if el is not None else np.NaN for el in X[col]], index=X[col].index)
            if self.handle_missing == "value":
                if not util.is_category(X[col].dtype):
                    X[col] = X[col].fillna(np.nan)

            if self._min_group_size is not None:
                if col in self._min_group_categories.keys():
                    X[col] = X[col].map(self._min_group_categories[col]).fillna(X[col])

            X[col] = X[col].astype(object).map(self.mapping[col])
            if isinstance(self._handle_unknown[col], (int, np.integer)):
                X[col] = X[col].fillna(self._handle_unknown[col])

            elif (self._handle_unknown[col] == 'value'
                  and X[col].isna().any()
                  and self._handle_missing[col] != 'return_nan'
            ):
                X[col].replace(np.nan, 0, inplace=True)

            elif (
                    self._handle_unknown[col] == 'error'
                    and X[col].isnull().any()
            ):
                raise ValueError(f'Missing data found in column {col} at transform time.')
        return X

    def _fit_count_encode(self, X_in, y):
        """Perform the count encoding."""
        X = X_in.copy(deep=True)

        if self.cols is None:
            self.cols = X.columns.values

        self.mapping = {}

        for col in self.cols:
            mapping_values = X[col].value_counts(normalize=self._normalize[col])
            ordinal_encoding = [m["mapping"] for m in self.ordinal_encoder.mapping if m["col"] == col][0]
            reversed_ordinal_enc = {v: k for k, v in ordinal_encoding.to_dict().items()}
            mapping_values.index = mapping_values.index.map(reversed_ordinal_enc)
            self.mapping[col] = mapping_values

            if self._handle_missing[col] == 'return_nan':
                self.mapping[col][np.NaN] = np.NaN
            
            # elif self._handle_missing[col] == 'value':
            #test_count.py failing     self.mapping[col].loc[-2] = 0

        if any([val is not None for val in self._min_group_size.values()]):
            self.combine_min_categories(X)

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
            pass
            # raise ValueError(
            #     "`combine_min_nan_groups` only works when `min_group_size` "
            #     "is set for all columns."
            # )

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
            'handle_unknown': 'value',
            'handle_missing': 'value',
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
                    f"'combine_min_nan_groups' == 'force' for columns `{col}`."
                )
            
            if (
                self._combine_min_nan_groups[col] is not True
                and self._min_group_size[col] is None
            ):
                raise ValueError(f"`combine_min_nan_groups` only works when `min_group_size` is set for column {col}.")

            if (
                self._min_group_name[col] is not None
                and self._min_group_size[col] is None
            ):
                raise ValueError(f"`min_group_name` only works when `min_group_size` is set for column {col}.")

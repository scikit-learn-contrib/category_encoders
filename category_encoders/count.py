"""Count Encoder"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util

__author__ = 'joshua t. dunn'


class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, cols=None, drop_invariant=False,
                 return_df=True, handle_unknown='value',
                 handle_missing='value',
                 min_group_size=None, combine_min_nan_groups=True,
                 min_group_name=None, normalize=False):
        """Count encoding for categorical features.

        For a given categorical feature, replace the names of the groups
        with the group counts.

        Parameters
        ----------

        verbose: int
            integer indicating verbosity of output. 0 for none.
        cols: list
            a list of columns to encode, if None, all string columns will be
            encoded.
        drop_invariant: bool
            boolean for whether or not to drop columns with 0 variance.
        return_df: bool
            boolean for whether to return a pandas DataFrame from transform
            (otherwise it will be a numpy array).
        handle_unknown: str
            how to handle unknown labels at transform time. Options are 'error'
            'return_nan' and 'value'. defaults to 'value' which uses  NaN behaviour
            specified at fit time or fills with 0 if no NaNs at fit time.

        handle_missing: str
            how to handle missing values at fit time. Options are 'error', 'return_nan',
            and 'value', default to 'value', which treat NaNs as a category at fit time.

        normalize: bool
            whether to normalize the counts to the range (0, 1). See Pandas `value_counts`
            for more details.
        min_group_size: int, float
            the minimun count threshold of a group needed to ensure it is not
            combined into a "leftovers" group. If float in the range (0, 1),
            `min_group_size` is calculated as int(X.shape[0] * min_group_size).
            Note: This value may change type based on the `normalize` variable. If True
            this will become a float. If False, it will be an int.
        min_group_name: None, str
            Set the name of the combined minimum groups when the defaults become
            to long. Default None. In this case the category names will be joined
            with a `_` delimeter.
        combine_min_nan_groups: bool
            whether to combine the leftovers group with NaN group. Default True. Can
            also be forced to combine with 'force' meaning small groups are effectively
            counted as NaNs. Only used when 'handle_missing' is value.


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
        if (
            isinstance(min_group_size, float)
            and (min_group_size >= 1.0)
            and (min_group_size <= 0.0)
        ):
            raise ValueError(
                'If `min_group_size` is float, '
                'it must be in the range (0, 1).'
            )

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
        self.count_nan_fit = handle_missing == 'value'
        self.min_group_size = min_group_size
        self.min_group_name = min_group_name
        self.combine_min_nan_groups = combine_min_nan_groups

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

        self.count_encode(X, y)

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

        if not self.cols:
            return X
        X, _ = self.count_encode(X, y)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def count_encode(self, X_in, y):
        """Perform the count encoding."""
        X = X_in.copy(deep=True)

        if self.cols is None:
            self.cols = X.columns.values

        if self.mapping is None:
            self.mapping = {}

            for col in self.cols:
                if (
                    self.handle_missing == 'error'
                    and X[col].isna().any()
                    ):

                    raise ValueError(
                        'Missing data found in column %s at fit time.'
                        % col
                    )

                self.mapping[col] = X[col].value_counts(
                    normalize=self.normalize,
                    dropna=not self.count_nan_fit
                )

            if self.min_group_size is not None:
                self.combine_min_categories(X)

        else:
            for col in self.cols:
                if self.min_group_size is not None:
                    if col in self._min_group_categories.keys():
                        X[col] = (
                            X[col].map(self._min_group_categories[col])
                            .fillna(X[col])
                        )

                X[col] = X[col].map(self.mapping[col])

                if self.handle_unknown == 'value':
                    X[col] = X[col].fillna(0)
                elif (
                    self.handle_unknown == 'error'
                    and X[col].isna().any()
                    ):

                    raise ValueError(
                        'Missing data found in column %s at transform time.'
                        % col
                    )

        return X, self.mapping

    def combine_min_categories(self, X):
        """Combine small categories into a single category."""
        if self.normalize and isinstance(self.min_group_size, int):
            self.min_group_size = self.min_group_size / X.shape[0]
        elif not self.normalize and isinstance(self.min_group_size, float):
            self.min_group_size = self.min_group_size * X.shape[0]

        self._min_group_categories = {}
        for col, mapper in self.mapping.items():
            if self.combine_min_nan_groups:
                min_groups_idx = mapper < self.min_group_size
            elif self.combine_min_nan_groups == 'force':
                min_groups_idx = (
                    (mapper < self.min_group_size)
                    & (mapper.index.isna())
                )
            else:
                min_groups_idx = (
                    (mapper < self.min_group_size)
                    & (~mapper.index.isna())
                )

            min_groups_sum = mapper.loc[min_groups_idx].sum()

            if min_groups_sum > 0 and (min_groups_idx).sum() > 1:
                if isinstance(self.min_group_name, str):
                    min_group_mapper_name = self.min_group_name
                else:
                    min_group_mapper_name = '_'.join([
                        str(idx)
                        for idx
                        in mapper.loc[min_groups_idx].index
                    ])
                
                self._min_group_categories[col] = {
                    cat: min_group_mapper_name
                    for cat
                    in mapper.loc[min_groups_idx].index.tolist()
                }

                mapper = mapper.loc[~min_groups_idx]
                
                if mapper.index.is_categorical():
                    mapper.index = mapper.index.add_categories(
                        min_group_mapper_name
                    )
                mapper[min_group_mapper_name] = min_groups_sum

            self.mapping[col] = mapper

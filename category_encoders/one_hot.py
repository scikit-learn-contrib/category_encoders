"""One-hot or dummy coding"""
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'willmcginnis'


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Onehot (or dummy) coding for categorical features, produces one feature per category, each binary.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    use_cat_names: bool
        if True, category values will be included in the encoded column names. Since this can result in duplicate column names, duplicates are suffixed with '#' symbol until a unique name is generated.
        If False, category indices will be used instead of the category values.
    handle_unknown: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
        an extra column will be added in if the transform matrix has unknown categories.  This can cause
        unexpected changes in dimension in some cases.
    handle_missing: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
        an extra column will be added in if the transform matrix has nan values.  This can cause
        unexpected changes in dimension in some cases.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OneHotEncoder(cols=['CHAS', 'RAD'], handle_unknown='indicator').fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 24 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS_1     506 non-null int64
    CHAS_2     506 non-null int64
    CHAS_-1    506 non-null int64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
    RAD_5      506 non-null int64
    RAD_6      506 non-null int64
    RAD_7      506 non-null int64
    RAD_8      506 non-null int64
    RAD_9      506 non-null int64
    RAD_-1     506 non-null int64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(13)
    memory usage: 95.0 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_missing='value', handle_unknown='value', use_cat_names=False):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.mapping = None
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.use_cat_names = use_cat_names
        self.feature_names = None

    @property
    def category_mapping(self):
        return self.ordinal_encoder.category_mapping

    def fit(self, X, y=None, **kwargs):
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

        # first check the type
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
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
        self.mapping = self.generate_mapping()

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

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

    def generate_mapping(self):
        mapping = []
        found_column_counts = {}

        for switch in self.ordinal_encoder.mapping:
            col = switch.get('col')
            values = switch.get('mapping').copy(deep=True)

            if self.handle_missing == 'value':
                values = values[values > 0]

            if len(values) == 0:
                continue

            index = []
            new_columns = []

            for cat_name, class_ in values.iteritems():
                if self.use_cat_names:
                    n_col_name = str(col) + '_%s' % (cat_name,)
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                else:
                    n_col_name = str(col) + '_%s' % (class_,)

                index.append(class_)
                new_columns.append(n_col_name)

            if self.handle_unknown == 'indicator':
                n_col_name = str(col) + '_%s' % (-1,)
                if self.use_cat_names:
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                new_columns.append(n_col_name)
                index.append(-1)

            base_matrix = np.eye(N=len(index), dtype=np.int)
            base_df = pd.DataFrame(data=base_matrix, columns=new_columns, index=index)

            if self.handle_unknown == 'value':
                base_df.loc[-1] = 0
            elif self.handle_unknown == 'return_nan':
                base_df.loc[-1] = np.nan

            if self.handle_missing == 'return_nan':
                base_df.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                base_df.loc[-2] = 0

            mapping.append({'col': col, 'mapping': base_df})

        return mapping

    def transform(self, X, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim, ))

        if not list(self.cols):
            return X if self.return_df else X.values

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.get_dummies(X)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data.

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

        # first check the type and make deep copy
        X = util.convert_input(X_in, columns=self.feature_names, deep=True)

        X = self.reverse_dummies(X, self.mapping)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "be False when transforming the data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (
                    X.shape[1], self._dim, ))

        if not list(self.cols):
            return X if self.return_df else X.values

        for switch in self.ordinal_encoder.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

            if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
                for col in self.cols:
                    if X[switch.get('col')].isnull().any():
                        warnings.warn("inverse_transform is not supported because transform impute "
                                      "the unknown category nan when encode %s" % (col,))

        return X if self.return_df else X.values

    def get_dummies(self, X_in):
        """
        Convert numerical variable into dummy variables

        Parameters
        ----------
        X_in: DataFrame

        Returns
        -------
        dummies : DataFrame

        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col])
            base_df = base_df.set_index(X.index)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        X = X.reindex(columns=cols)

        return X

    def reverse_dummies(self, X, mapping):
        """
        Convert dummy variable into numerical variables

        Parameters
        ----------
        X : DataFrame
        mapping: list-like
              Contains mappings of column to be transformed to it's new columns and value represented

        Returns
        -------
        numerical: DataFrame

        """
        out_cols = X.columns.values.tolist()
        mapped_columns = []
        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            insert_at = out_cols.index(mod.columns[0])

            X.insert(insert_at, col, 0)
            positive_indexes = mod.index[mod.index > 0]
            for i in range(positive_indexes.shape[0]):
                existing_col = mod.columns[i]
                val = positive_indexes[i]
                X.loc[X[existing_col] == 1, col] = val
                mapped_columns.append(existing_col)
            X.drop(mod.columns, axis=1, inplace=True)
            out_cols = X.columns.values.tolist()

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
                'Must transform data first. Affected feature names are not known before.')
        else:
            return self.feature_names

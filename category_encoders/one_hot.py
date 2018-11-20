"""One-hot or dummy coding"""
import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'willmcginnis'


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Onehot (or dummy) coding for categorical features, produces one feature per category, each binary.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1. Warning: if
        impute is used, an extra column will be added in if the transform matrix has unknown categories. This can cause
        unexpected changes in the dimension in some cases.
    use_cat_names: bool
        if True, category values will be included in the encoded column names. Since this can result into duplicate column names, duplicates are suffixed with '#' symbol until a unique name is generated.
        If False, category indices will be used instead of the category values.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OneHotEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 24 columns):
    CHAS_1     506 non-null int64
    CHAS_2     506 non-null int64
    CHAS_-1    506 non-null int64
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
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(13)
    memory usage: 95.0 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True, handle_unknown='impute', use_cat_names=False):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.mapping = None
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
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

        # Indicate no transformation has been applied yet

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
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
            column_mapping = switch.get('mapping').copy(deep=True)

            if self.handle_unknown == 'impute':
                column_mapping = column_mapping.append(pd.Series(data=[-1], index=['-1']))

            col_mappings = []
            for cat_name, class_ in column_mapping.iteritems():
                if self.use_cat_names:
                    n_col_name = str(col) + '_%s' % (cat_name,)
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                else:
                    n_col_name = str(col) + '_%s' % (class_,)
                col_mappings.append({'new_col_name': n_col_name, 'val': class_})

            mapping.append({'col': col, 'mapping': col_mappings})
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

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim, ))

        if not self.cols:
            return X if self.return_df else X.values

        X = self.ordinal_encoder.transform(X)

        X = self.get_dummies(X, mapping=self.mapping)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        # Now we can build the list of new / transformed columns
        self.feature_names = X.columns

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
        X = X_in.copy(deep=True)

        # first check the type
        X = util.convert_input(X)

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to inverse_transform data')

        X = self.reverse_dummies(X, self.mapping)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "set as False when transform data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (
                    X.shape[1], self._dim, ))

        if not self.cols:
            return X if self.return_df else X.values

        if self.impute_missing and self.handle_unknown == 'impute':
            for col in self.cols:
                if any(X[col] == -1):
                    raise ValueError("inverse_transform is not supported because transform impute "
                                     "the unknown category -1 when encode %s"%(col,))

        for switch in self.ordinal_encoder.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.get_values())
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X if self.return_df else X.values

    def get_dummies(self, X_in, mapping):
        """
        Convert numerical variable into dummy variables

        Parameters
        ----------
        X_in: DataFrame
        mapping: list-like
              Contains mappings of column to be transformed to it's new columns and value represented

        Returns
        -------
        dummies : DataFrame
        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            new_columns = []
            for column_mapping in mod:
                new_col_name = column_mapping['new_col_name']
                val = column_mapping['val']
                X[new_col_name] = (X[col] == val).astype(int)
                new_columns.append(new_col_name)
            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = new_columns

        return X.reindex(columns=cols)

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
        out_cols = X.columns.values
        cols = []
        mapped_columns = []
        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            cols.append(col)

            X[col] = 0
            for column_mapping in mod:
                existing_col = column_mapping.get('new_col_name')
                val = column_mapping.get('val')
                X.loc[X[existing_col] == 1, col] = val
                mapped_columns.append(existing_col)

        out_cols = [col0 for col0 in out_cols if col0 not in mapped_columns]

        return X.reindex(columns=out_cols + cols)

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        --------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!
        """

        if not isinstance(self.feature_names, list):
            raise ValueError(
                'Must transform data first. Affected feature names are not known before.')
        else:
            return self.feature_names

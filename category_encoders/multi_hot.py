"""Multi-hot or dummy coding"""
import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'fullflu'


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Multihot coding for categorical features, produces one feature per category, each non-negative.

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

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True, handle_unknown='impute', use_cat_names=False, multiple_split_string="|"):
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
        self.multiple_split_string = multiple_split_string

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
        X = util.convert_input(X).fillna(np.nan)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        # Indicate no transformation has been applied yet
        self.mapping, self.col_index_mapping = self.generate_mapping(X, cols=self.cols, multiple_split_string=self.multiple_split_string)

        return self

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

    @staticmethod
    def generate_mapping(X_in, mapping=None, cols=None, impute_missing=True, handle_unknown='impute', multiple_split_string="|"):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.
        """
        X = X_in.copy(deep=True)
        # dictionary which maps col name to index of ordinal mapping
        col_index_mapping = {}

        if cols is None:
            cols = X.columns.values

        mapping_out = []
        for i, col in enumerate(cols):
            # append index mapping
            col_index_mapping[col] = i
            candidate_of_col = set()
            tmp = (X[col].map(lambda x: candidate_of_col.update(x.split(multiple_split_string))
                if type(x) == str
                else candidate_of_col.add(str(int(x)))
                if (type(x) == float or type(x) == int) and x == x
                else candidate_of_col.add(x)))

            col_mappings = []
            # val to new_colname mapping
            val_newcolname_mappings = {}
            for i, class_ in enumerate(candidate_of_col):
                n_col_name = str(col) + '_%s' % (i + 1,)
                col_mappings.append({'new_col_name': n_col_name, 'val': class_})
                val_newcolname_mappings[class_] = n_col_name

            mapping_out.append({'col': col, 'mapping': col_mappings, 'data_type': X[col].dtype, 'candidate_set': candidate_of_col, "val_newcolname_mapping": val_newcolname_mappings})

        return mapping_out, col_index_mapping

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
        X = util.convert_input(X).fillna(np.nan)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[1], self._dim, ))

        if not self.cols:
            return X if self.return_df else X.values

        X = self.get_dummies(X, mapping=self.mapping, multiple_split_string=self.multiple_split_string)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def get_dummies(self, X_in, mapping, multiple_split_string="|"):
        """
        Convert numerical variable into dummy variables
        Parameters
        ----------
        X_in: DataFrame
        mapping: list-like
              Contains mappings of column to be transformed to it's new columns and value represented
        multiple: Boolean
              Represents to user multi-hot encoder or not
        Returns
        -------
        dummies : DataFrame
        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            inv_map = switch.get('val_newcolname_mapping')
            new_columns = []
            # for column_mapping in mod:
            #     new_col_name = column_mapping['new_col_name']
            #     X[new_col_name] = 0
            for column_mapping in mod:
                new_col_name = column_mapping['new_col_name']
                val = column_mapping['val']
                # multi-hot encoder
                X[new_col_name] = (X[col].map(lambda x: 1
                    if (type(x) == str and val in x.split(self.multiple_split_string))
                    or (val != val and x != x)
                    or ((type(x) == int or type(x) == float) and x == x and val == str(int(x)))
                    else 0))
                new_columns.append(new_col_name)
            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = list(set(new_columns))

        return X.reindex(columns=cols)

"""On-the-fly frequency coding"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

__author__ = 'Guolin Ke'

class OnFlyFrequencyEncoder(BaseEstimator, util.TransformerWithTargetMixin):
    """OnFlyFrequency coding for categorical features.

    This is very similar to catboost encoding, but is unsupervised. 
    This encoder calculates the frequency of each category "on-the-fly". Consequently, the values naturally vary
    during the training phase and it is not necessary to add random noise.

    Beware, the training data have to be randomly permutated. E.g.:

        # Random permutation
        perm = np.random.permutation(len(X))
        X = X.iloc[perm].reset_index(drop=True)
        y = y.iloc[perm].reset_index(drop=True)

    This is necessary because some data sets are sorted based on the target
    value and this coder encodes the features on-the-fly in a single pass.

    Beware, although this is an unsupervised encoding, you need to pass 'y' in 'fit' and 'transform' when encoding training data.
    'y' is not used in encoding, it is just used for distinguish the training data.

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
    handle_missing: str
        options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = OnFlyFrequencyEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS       506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.use_default_cols = cols is None  # if True, even a repeated call of fit() will select string columns from X
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.feature_names = None

    def fit(self, X, y, **kwargs):
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

        # unite the input into pandas types
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.use_default_cols:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        mapping = self._fit(
            X,
            cols=self.cols
        )
        self.mapping = mapping

        X_temp = self.transform(X, y, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

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

    def transform(self, X, y=None, override_return_df=False):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform without target information (such as transform test set)

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # unite the input into pandas types
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))


        if not list(self.cols):
            return X
        X = self._transform(
            X,
            y,
            mapping=self.mapping
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    def _fit(self, X_in, cols=None):
        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values
        mapping = {}
        for col in cols:
            col_series = X[col]
            category = pd.Categorical(col_series)
            cnt_cats = len(category.categories)
            result = category.value_counts(dropna=False) / len(col_series)
            mapping[col] = {'freq': result, 'prior': 1.0 / cnt_cats}
        return mapping

    def _transform(self, X_in, y=None, mapping=None):
        """
        The model uses a single column of floats to represent the means of the target variables.
        """
        X = X_in.copy(deep=True)

        for col, colmap in mapping.items():
            prior = colmap['prior']
            unique_train = colmap['freq'].index
            is_nan = X_in[col].isnull()
            # don't include nan in unknown
            is_unknown_value = ~X_in[col].isin(unique_train) & (~is_nan)

            if self.handle_unknown == 'error' and is_unknown_value.any():
                raise ValueError('Columns to be encoded can not contain new values')

            if y is None:
                X[col] = X[col].map(colmap['freq'])
            else:
                temp = y.groupby(X[col].astype(str)).agg(['cumcount']) + 1
                total_cnt = pd.Series(range(1,len(y) + 1))
                X[col] = (temp['cumcount'] / total_cnt)

            if self.handle_unknown == 'value':
                if X[col].dtype.name == 'category':
                    X[col] = X[col].astype(float)
                X.loc[is_unknown_value, col] = prior
            elif self.handle_unknown == 'return_nan':
                X.loc[is_unknown_value, col] = np.nan
            
            # only fill nan when it is not in train
            if self.handle_missing == 'value' and np.nan not in unique_train:
                X.loc[is_nan, col] = prior
            elif self.handle_missing == 'return_nan':
                X.loc[is_nan, col] = np.nan

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
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names

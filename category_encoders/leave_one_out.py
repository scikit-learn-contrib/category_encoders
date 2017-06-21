"""Leave one out coding"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import get_obj_cols, convert_input
from sklearn.utils.random import check_random_state

__author__ = 'hbghhy'


class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """Leave one out coding for categorical features.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array)
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1. Warning: if
        impute is used, an extra column will be added in if the transform matrix has unknown categories.  This can causes
        unexpected changes in dimension in some cases.

    Example
    -------
    >>>from category_encoders import *
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>enc = LeaveOneOutEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>>numeric_dataset = enc.transform(X)
    >>>print(numeric_dataset.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 22 columns):
    CHAS_0     506 non-null int64
    CHAS_1     506 non-null int64
    RAD_0      506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
    RAD_5      506 non-null int64
    RAD_6      506 non-null int64
    RAD_7      506 non-null int64
    RAD_8      506 non-null int64
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
    dtypes: float64(11), int64(11)
    memory usage: 87.0 KB
    None

    References
    ----------

    .. [1] Strategies to encode categorical variables with many categories. from
    https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154.



    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True,
                 handle_unknown='impute'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self._mean = None

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

        # first check the type
        X = convert_input(X)
        y = pd.Series(y, name='target')
        assert X.shape[0] == y.shape[0]

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        _, categories = self.leave_one_out(
            X, y,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        self.mapping = categories

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X, y=None, random_mized=False, random_state=None, sigma=0.05):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform withour target infor(such as transform test set)
        random_mized : boolean, Add normal (Gaussian) distribution randomized to the encoder or not
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        sigma : float or array_like of floats, Standard deviation (spread or "width") of the distribution.
            

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))
        assert (y is None or X.shape[0] == y.shape[0])
        if isinstance(sigma, np.ndarray):
            assert (sigma.shape[0] == X.shape[0])

        if not self.cols:
            return X
        X, _ = self.leave_one_out(
            X, y,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown,
            random_mized=random_mized,
            random_state=random_state,
            sigma=sigma
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def leave_one_out(self, X_in, y, mapping=None, cols=None, impute_missing=True, handle_unknown='impute',
                      random_mized=False, random_state=None, sigma=0.1):
        """
        Leave one out encoding uses a single column of float to represent the mean of targe variable.
        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            if random_mized:
                gen = check_random_state(random_state)
            for switch in mapping:
                X[str(switch.get('col')) + '_tmp'] = np.nan
                for val in switch.get('mapping'):
                    if y is None:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = \
                            switch.get('mapping')[val]['mean']
                    elif switch.get('mapping')[val]['count'] == 1:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = self._mean
                    else:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = (
                            (switch.get('mapping')[val]['sum'] - y[(X[switch.get('col')] == val).values]) / (
                                switch.get('mapping')[val]['count'] - 1)
                        )
                del X[switch.get('col')]
                X.rename(columns={str(switch.get('col')) + '_tmp': switch.get('col')}, inplace=True)

                if impute_missing:
                    if handle_unknown == 'impute':
                        X[switch.get('col')].fillna(self._mean, inplace=True)
                    elif handle_unknown == 'error':
                        if X[~X['D'].isin([str(x[1]) for x in switch.get('mapping')])].shape[0] > 0:
                            raise ValueError('Unexpected categories found in %s' % (switch.get('col'),))

                if random_mized:
                    X[switch.get('col')] = X[switch.get('col')] * gen.normal(1., sigma, X[switch.get('col')].shape[0])

                X[switch.get('col')] = X[switch.get('col')].astype(float).values.reshape(-1, )
        else:
            self._mean = y.mean()
            mapping_out = []

            for col in cols:
                tmp = y.groupby(X[col]).agg(['sum', 'count'])
                tmp['mean'] = tmp['sum'] / tmp['count']
                tmp = tmp.to_dict(orient='index')

                X[str(col) + '_tmp'] = np.nan
                for val in tmp:
                    """if the val only appear once ,encoder it as mean of y"""
                    if tmp[val]['count'] == 1:
                        X.loc[X[col] == val, str(col) + '_tmp'] = self._mean
                    else:
                        X.loc[X[col] == val, str(col) + '_tmp'] = (tmp[val]['sum'] - y.loc[X[col] == val]) / (
                            tmp[val]['count'] - 1)
                del X[col]
                X.rename(columns={str(col) + '_tmp': col}, inplace=True)

                if impute_missing:
                    if handle_unknown == 'impute':
                        X[col].fillna(self._mean, inplace=True)

                X[col] = X[col].astype(float).values.reshape(-1, )

                mapping_out.append({'col': col, 'mapping': tmp}, )

        return X, mapping_out

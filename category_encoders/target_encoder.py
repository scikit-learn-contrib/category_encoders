"""Target Encoder"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import get_obj_cols, convert_input
from sklearn.utils.random import check_random_state

__author__ = 'chappers'


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, impute_missing=True,
                 handle_unknown='impute', min_samples_leaf=1, smoothing=1):
        """Target Encode for categorical features. Based on leave one out approach.

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

    min_samples_leaf : int
        minimum samples to take category average into account
    smoothing : int
        smoothing effect to balance categorical average vs prior

    Example
    -------
    >>>from category_encoders import *
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>enc = TargetEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>>numeric_dataset = enc.transform(X)
    >>>print(numeric_dataset.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
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
    CHAS       506 non-null float64
    RAD        506 non-null float64
    dtypes: float64(13)
    memory usage: 51.5 KB
    None

    References
    ----------

    .. [1] A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems. from
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf.
        """
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
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
        _, categories = self.target_encode(
            X, y,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown,
            smoothing_in=self.smoothing,
            min_samples_leaf=self.min_samples_leaf
        )
        self.mapping = categories

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X, y=None):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform withour target infor(such as transform test set)
            
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

        if not self.cols:
            return X
        X, _ = self.target_encode(
            X, y,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown, 
        )

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values
    
    def target_encode(self, X_in, y, mapping=None, cols=None, impute_missing=True,
            handle_unknown='impute', min_samples_leaf=1, smoothing_in=1):
        X = X_in.copy(deep=True)
        if cols is None:
            cols = X.columns.values
        
        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                X[str(switch.get('col')) + '_tmp'] = np.nan
                for val in switch.get('mapping'):
                    if switch.get('mapping')[val]['count'] == 1:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = self._mean
                    else:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = \
                            switch.get('mapping')[val]['smoothing']
                del X[switch.get('col')]
                X.rename(columns={str(switch.get('col')) + '_tmp': switch.get('col')}, inplace=True)

                if impute_missing:
                    if handle_unknown == 'impute':
                        X[switch.get('col')].fillna(self._mean, inplace=True)
                    elif handle_unknown == 'error':
                        if X[~X[switch.get('col')].isin([str(x[1]) for x in switch.get('mapping')])].shape[0] > 0:
                            raise ValueError('Unexpected categories found in %s' % (switch.get('col'),))

                X[switch.get('col')] = X[switch.get('col')].astype(float).values.reshape(-1, )
        
        else:
            self._mean = y.mean()
            prior = self._mean
            mapping_out = []
            for col in cols:
                tmp = y.groupby(X[col]).agg(['sum', 'count'])
                tmp['mean'] = tmp['sum'] / tmp['count']
                tmp = tmp.to_dict(orient='index')

                X[str(col) + '_tmp'] = np.nan
                for val in tmp:
                    if tmp[val]['count'] == 1:
                        X.loc[X[col] == val, str(col) + '_tmp'] = self._mean
                    else:
                        # Make smoothing a float so that python 2 does not treat as integer division
                        smoothing = float(smoothing_in)
                        smoothing = 1 / (1 + np.exp(-(tmp[val]["count"] - min_samples_leaf) / smoothing))
                        cust_smoothing = prior * (1 - smoothing) + tmp[val]['mean'] * smoothing
                        X.loc[X[col] == val, str(col) + '_tmp'] = cust_smoothing
                        tmp[val]['smoothing'] = cust_smoothing
                del X[col]
                X.rename(columns={str(col) + '_tmp': col}, inplace=True)
                if impute_missing:
                    if handle_unknown == 'impute':
                        X[col].fillna(self._mean, inplace=True)

                X[col] = X[col].astype(float).values.reshape(-1, )

                mapping_out.append({'col': col, 'mapping': tmp}, )

        return X, mapping_out

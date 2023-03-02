import numpy as np
import pandas as pd
import category_encoders.utils as util

__author__ = 'Khaled_Issa'


class ResponseCoding(util.BaseEncoder, util.SupervisedTransformerMixin):
    """
    Response coding for the encoding of categorical features.

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
        options are 'error', 'return_nan'  and 'value', defaults to 'value'.
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value'.
    

    Examples
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = ResponseCoding(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 9 columns):
    #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
    0   Id            1460 non-null   float64
    1   MSSubClass    1460 non-null   float64
    2   MSZoning      1460 non-null   object 
    3   LotFrontage   1201 non-null   float64
    4   YearBuilt     1460 non-null   float64
    5   CentralAir_0  0 non-null      float64
    6   CentralAir_1  0 non-null      float64
    7   Heating_0     1460 non-null   float64
    8   Heating_1     1460 non-null   float64
    dtypes: float64(8), object(1)
    memory usage: 102.8+ KB
    None
   
    References
    ----------

    https://medium.com/@thewingedwolf.winterfell/response-coding-for-categorical-data-7bb8916c6dc1
    
    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_N_UNIQUE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value',
                 handle_unknown='value'):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)        
        self.mapping = None

    
    def _fit(self, X, y, **kwargs):
        self.mapping = self.fit_response_coding(X, y)
        return self
    
    def fit_response_coding(self, X, y):
        mapping = {1: {}, 0: {}}
        for col in self.cols:            
            for label in [0,1]:
                mapping[label] = dict((X[col][y == label].value_counts() / X[col].value_counts()).fillna(0))
        return mapping
   
    def _transform(self, X, y=None):
        X = self.response_code(X)
        return X

    def response_code(self, X_in):
        X = X_in.copy(deep=True)
       
        for col in self.cols:
            X[col + '_0'] = X[col].map(self.mapping[0])
            X[col + '_1'] = X[col].map(self.mapping[1])
            X.drop(col, axis=1, inplace = True)
        return X
        
        
    
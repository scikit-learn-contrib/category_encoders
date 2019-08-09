"""Target Encoder with smoothing"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import category_encoders.utils as util

__author__ = 'ankur_singh'

class TargetEncodingSmoothing(BaseEstimator, TransformerMixin):
    """Target encoding (with smoothing) for categorical features.
    
    Makes target encoding more robust to leakage by addressing main problem - low-frequency values. If in your feature there are unique values which occurs just couple of times - they are one of the main source of leak.

What if instead of encoding by mean(incase of Target Encoding) we will take weighted sum of 2 means: dataset mean and level mean, where level mean is the mean of particular unique value in your feature.

    Parameters
    ----------

    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    k: float
        inflection point, that's the point where  𝑓(𝑥)  is equal 0.5
    f: float
        steepness, a value which controls how step is our function.
    
    Example-1: Using Target Encoding with Smoothing
    -----------------------------------------------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> mushroom = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', header=None)
    >>> y = mushroom.iloc[:,0]
    >>> X = mushroom.iloc[:,1:]
    >>> enc = TargetEncodingSmoothing().fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 22 columns):
    1     8124 non-null float64
    2     8124 non-null float64
    3     8124 non-null float64
    4     8124 non-null float64
    5     8124 non-null float64
    6     8124 non-null float64
    7     8124 non-null float64
    8     8124 non-null float64
    9     8124 non-null float64
    10    8124 non-null float64
    11    8124 non-null float64
    12    8124 non-null float64
    13    8124 non-null float64
    14    8124 non-null float64
    15    8124 non-null float64
    16    8124 non-null float64
    17    8124 non-null float64
    18    8124 non-null float64
    19    8124 non-null float64
    20    8124 non-null float64
    21    8124 non-null float64
    22    8124 non-null float64
    dtypes: float64(22)
    memory usage: 1.4 MB
    None
    
    Example-2: Visualizing Smoothing Curves
    ---------------------------------------
    >>> x = np.linspace(0,100,100)
    >>> plot = pd.DataFrame()
    >>> te = TargetEncodingSmoothing([], 1,1)
    >>> plot["k=1|f=1"] = te.smoothing_func(x)
    >>> te = TargetEncodingSmoothing([], 33,5)
    >>> plot["k=33|f=5"] = te.smoothing_func(x)
    >>> te = TargetEncodingSmoothing([], 66,15)
    >>> plot["k=66|f=15"] = te.smoothing_func(x)
    >>> plot.plot(figsize = (15,8))

    References
    ----------

    ..[1] Kaggle kernel for Employee Access Challenge, from
    https://www.kaggle.com/dmitrylarko/kaggledays-sf-3-amazon-supervised-encoding

    """

    def __init__(self, cols = None, k=3, f=1.5):
        self.columns_names = cols
        self.learned_values = {}
        self.dataset_mean = np.nan
        self.k = k 
        self.f = f 
        self._dim = None
        
    def smoothing_func(self, N): #
        return 1 / (1 + np.exp(-(N-self.k)/self.f))
   
    def fit(self, X, y, **fit_params):
        """ Fit encoder according to X and y.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vales.
            
        Returns
        -------
        self : encoder
            Return self.
        
        """
        
        # unite the input into pandas types
        X = util.convert_input(X)
        y = util.convert_input_vector(y, X.index)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("The lenght of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")
        
        self._dim = X.shape[1]
        
        # if columns aren't passed, just use every string column
        if self.columns_names is None:
            self.columns_names = util.get_obj_cols(X)
        else:
            self.columns_names = util.convert_cols_to_list(self.columns_names)
        
        X_ = X.copy()
        self.learned_values = {}
        if y.dtype == object:
            _y = LabelEncoder().fit_transform(y)
        self.dataset_mean = np.mean(_y)
        X_["__target__"] = _y
        
        for c in [x for x in X_.columns if x in self.columns_names]:
            stats = (X_[[c,"__target__"]]
                     .groupby(c)["__target__"].
                     agg(['mean', 'size'])) 
            stats["alpha"] = self.smoothing_func(stats["size"])
            stats["__target__"] = (stats["alpha"]*stats["mean"] 
                                   + (1-stats["alpha"])*self.dataset_mean)
            stats = (stats
                     .drop([x for x in stats.columns if x not in ["__target__",c]], axis = 1)
                     .reset_index())
            self.learned_values[c] = stats
        self.dataset_mean = np.mean(_y)
        return self
    
    def transform(self, X, **fit_params):
        """ Perform the transformation to new categorical Data.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_sampls]
        
        Returns:
        --------
        p : array, shape = [n_samples, n_features]
            Transformed values with encoding applied.
        
        """
        
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transfrom data.')
        
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))
        
        transformed_X = X.copy()
        for c in self.columns_names:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on = c, how = 'left')
                               )["__target__"]
        transformed_X[self.columns_names] = transformed_X[self.columns_names].fillna(self.dataset_mean)
        return transformed_X
    
    def fit_transform(self, X, y, **fit_params):
        """ Perform the training plus transformation to categorical Data.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_sampls]
        
        Returns:
        --------
        p : array, shape = [n_samples, n_features]
            Transformed values with encoding applied.
        
        """
        self.fit(X,y, **fit_params)
        return self.transform(X)
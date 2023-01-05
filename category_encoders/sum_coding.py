"""Sum contrast coding"""


from patsy.contrasts import ContrastMatrix, Sum
import numpy as np

from category_encoders.base_contrast_encoder import BaseContrastEncoder

__author__ = 'paulwestenthanner'

class SumEncoder(BaseContrastEncoder):
    """Sum contrast coding for the encoding of categorical features.

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
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = SumEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   intercept     1460 non-null   int64  
     1   Id            1460 non-null   float64
     2   MSSubClass    1460 non-null   float64
     3   MSZoning      1460 non-null   object 
     4   LotFrontage   1201 non-null   float64
     5   YearBuilt     1460 non-null   float64
     6   Heating_0     1460 non-null   float64
     7   Heating_1     1460 non-null   float64
     8   Heating_2     1460 non-null   float64
     9   Heating_3     1460 non-null   float64
     10  Heating_4     1460 non-null   float64
     11  CentralAir_0  1460 non-null   float64
    dtypes: float64(10), int64(1), object(1)
    memory usage: 137.0+ KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf

    """

    def get_contrast_matrix(self, values_to_encode: np.array) -> ContrastMatrix:
        return Sum().code_without_intercept(values_to_encode.tolist())

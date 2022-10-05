"""Helmert contrast coding"""


from patsy.contrasts import ContrastMatrix, Helmert
import numpy as np

from category_encoders.base_contrast_encoder import BaseContrastEncoder

__author__ = 'paulwestenthanner'


class HelmertEncoder(BaseContrastEncoder):
    """Helmert contrast coding for encoding categorical features.

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
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = HelmertEncoder(cols=['CHAS', 'RAD'], handle_unknown='value', handle_missing='value').fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 21 columns):
    intercept    506 non-null int64
    CRIM         506 non-null float64
    ZN           506 non-null float64
    INDUS        506 non-null float64
    CHAS_0       506 non-null float64
    NOX          506 non-null float64
    RM           506 non-null float64
    AGE          506 non-null float64
    DIS          506 non-null float64
    RAD_0        506 non-null float64
    RAD_1        506 non-null float64
    RAD_2        506 non-null float64
    RAD_3        506 non-null float64
    RAD_4        506 non-null float64
    RAD_5        506 non-null float64
    RAD_6        506 non-null float64
    RAD_7        506 non-null float64
    TAX          506 non-null float64
    PTRATIO      506 non-null float64
    B            506 non-null float64
    LSTAT        506 non-null float64
    dtypes: float64(20), int64(1)
    memory usage: 83.1 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf

    """
    def get_contrast_matrix(self, values_to_encode: np.array) -> ContrastMatrix:
        return Helmert().code_without_intercept(values_to_encode)

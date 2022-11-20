"""Gray encoding"""
from functools import partialmethod

import pandas as pd

from category_encoders import utils
from category_encoders.basen import BaseNEncoder
from typing import List

__author__ = 'paulwestenthanner'


class GrayEncoder(BaseNEncoder):
    """Gray encoding for categorical variables.
    Gray encoding is a form of binary encoding where consecutive values only differ by a single bit.
    Hence, gray encoding only makes sense for ordinal features.
    This has benefits in privacy preserving data publishing.

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
    >>> from category_encoders import GrayEncoder
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names_out_)
    >>> enc = GrayEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 18 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS_0     506 non-null int64
    CHAS_1     506 non-null int64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD_0      506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(7)
    memory usage: 71.3 KB
    None

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Gray_code
    .. [2] Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. 2017. PrivBayes:
    Private Data Release via Bayesian Networks. ACM Trans. Database Syst. 42, 4, Article 25 (October 2017)
    """
    encoding_relation = utils.EncodingRelation.ONE_TO_M
    __init__ = partialmethod(BaseNEncoder.__init__, base=2)

    @staticmethod
    def gray_code(n, n_bit) -> List[int]:
        gray = n ^ (n >> 1)
        gray_formatted = "{0:0{1}b}".format(gray, n_bit)
        return [int(bit) for bit in gray_formatted]

    def _fit(self, X, y=None, **kwargs):
        super(GrayEncoder, self)._fit(X, y, **kwargs)
        gray_mapping = []
        # convert binary mapping to Gray mapping and reorder
        for col_to_encode in self.mapping:
            col = col_to_encode["col"]
            bin_mapping = col_to_encode["mapping"]
            n_cols_out = bin_mapping.shape[1]
            null_cond = (bin_mapping.index < 0) | (bin_mapping.isnull().all(1))
            map_null = bin_mapping[null_cond]
            map_non_null = bin_mapping[~null_cond].copy()
            ordinal_mapping = [m for m in self.ordinal_encoder.mapping if m.get("col") == col]
            if len(ordinal_mapping) != 1:
                raise ValueError("Cannot find ordinal encoder mapping of Gray encoder")
            ordinal_mapping = ordinal_mapping[0]["mapping"]
            reverse_ordinal_mapping = {v: k for k, v in ordinal_mapping.to_dict().items()}
            map_non_null["orig_value"] = map_non_null.index.to_series().map(reverse_ordinal_mapping)
            map_non_null = map_non_null.sort_values(by="orig_value")
            gray_encoding = [self.gray_code(i + 1, n_cols_out) for i in range(map_non_null.shape[0])]
            gray_encoding = pd.DataFrame(data=gray_encoding, index=map_non_null.index, columns=bin_mapping.columns)
            gray_encoding = pd.concat([gray_encoding, map_null])
            gray_mapping.append({"col": col, "mapping": gray_encoding})
        self.mapping = gray_mapping

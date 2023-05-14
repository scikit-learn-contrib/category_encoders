"""Ordinal or label encoding"""

import numpy as np
import pandas as pd
import category_encoders.utils as util
import warnings
from typing import Dict, List, Union

__author__ = 'willmcginnis'


class OrdinalEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):
    """Encodes categorical features as ordinal, in one ordered feature.

    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in; in this case, we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

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
    mapping: list of dicts
        a mapping of class to label to use for the encoding, optional.
        the dict contains the keys 'col' and 'mapping'.
        the value of 'col' should be the feature name.
        the value of 'mapping' should be a dictionary or pd.Series of 'original_label' to 'encoded_label'.
        example mapping: [
            {'col': 'col1', 'mapping': {None: 0, 'a': 1, 'b': 2}},
            {'col': 'col2', 'mapping': {None: 0, 'x': 1, 'y': 2}}
        ]
    handle_unknown: str
        options are 'error', 'return_nan' and 'value', defaults to 'value', which will impute the category -1.
    handle_missing: str
        options are 'error', 'return_nan', and 'value, default to 'value', which treat nan as a category at fit time,
        or -2 at transform time if nan is not a category during fit.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = OrdinalEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Id           1460 non-null   float64
     1   MSSubClass   1460 non-null   float64
     2   MSZoning     1460 non-null   object 
     3   LotFrontage  1201 non-null   float64
     4   YearBuilt    1460 non-null   float64
     5   Heating      1460 non-null   int64  
     6   CentralAir   1460 non-null   int64  
    dtypes: float64(4), int64(2), object(1)
    memory usage: 80.0+ KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf

    """
    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, mapping=None, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.mapping_supplied = mapping is not None
        if self.mapping_supplied:
            mapping = self._validate_supplied_mapping(mapping)
        self.mapping = mapping

    @property
    def category_mapping(self):
        return self.mapping

    def _fit(self, X, y=None, **kwargs):
        # reset mapping in case of refit
        if not self.mapping_supplied:
            self.mapping = None
        _, categories = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing
        )
        self.mapping = categories

    def _transform(self, X):

        X, _ = self.ordinal_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing
        )
        return X

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data. Will attempt best case reconstruction, which means
        it will return nan for handle_missing and handle_unknown settings that break the bijection. We issue
        warnings when some of those cases occur.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """

        # fail fast
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        # first check the type and make deep copy
        X = util.convert_input(X_in, deep=True)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError(f"Unexpected input dimension {X.shape[1]}, the attribute drop_invariant should "
                                 "be False when transforming the data")
            else:
                raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')

        if not list(self.cols):
            return X if self.return_df else X.values

        if self.handle_unknown == 'value':
            for col in self.cols:
                if any(X[col] == -1):
                    warnings.warn("inverse_transform is not supported because transform impute "
                                  f"the unknown category -1 when encode {col}")

        if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
            for col in self.cols:
                if X[col].isnull().any():
                    warnings.warn("inverse_transform is not supported because transform impute "
                                  f"the unknown category nan when encode {col}")

        for switch in self.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X if self.return_df else X.values

    @staticmethod
    def ordinal_encoding(X_in, mapping=None, cols=None, handle_unknown='value', handle_missing='value'):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.
        """

        return_nan_series = pd.Series(data=[np.nan], index=[-2])

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                column = switch.get('col')
                col_mapping = switch['mapping']

                # Treat None as np.nan
                X[column] = pd.Series([el if el is not None else np.NaN for el in X[column]], index=X[column].index)
                X[column] = X[column].map(col_mapping)
                if util.is_category(X[column].dtype):
                    nan_identity = col_mapping.loc[col_mapping.index.isna()].values[0]
                    X[column] = X[column].cat.add_categories(nan_identity)
                    X[column] = X[column].fillna(nan_identity)
                try:
                    X[column] = X[column].astype(int)
                except ValueError as e:
                    X[column] = X[column].astype(float)

                if handle_unknown == 'value':
                    X[column].fillna(-1, inplace=True)
                elif handle_unknown == 'error':
                    missing = X[column].isnull()
                    if any(missing):
                        raise ValueError(f'Unexpected categories found in column {column}')

                if handle_missing == 'return_nan':
                    X[column] = X[column].map(return_nan_series).where(X[column] == -2, X[column])

        else:
            mapping_out = []
            for col in cols:
                nan_identity = np.nan
                categories = X[col].unique()
                # make nan last category
                if pd.isna(categories).any():
                    categories = [c for c in categories if not pd.isna(c)] + [nan_identity]
                else:
                    categories = categories.tolist()
                if util.is_category(X[col].dtype):
                    # Avoid using pandas category dtype meta-data if possible, see #235, #238.
                    if X[col].dtype.ordered:
                        categories = list(set(categories).intersection(set(X[col].dtype.categories)))
                    if X[col].isna().any():
                        categories += [np.nan]

                index = pd.Series(categories).fillna(nan_identity).unique()

                data = pd.Series(index=index, data=range(1, len(index) + 1))

                if handle_missing == 'value' and ~data.index.isnull().any():
                    data.loc[nan_identity] = -2
                elif handle_missing == 'return_nan':
                    data.loc[nan_identity] = -2

                mapping_out.append({'col': col, 'mapping': data, 'data_type': X[col].dtype}, )

        return X, mapping_out

    def _validate_supplied_mapping(self, supplied_mapping: List[Dict[str, Union[str, Dict, pd.Series]]]) -> List[Dict[str, Union[str, pd.Series]]]:
        """
        validate the supplied mapping and convert the actual mapping per column to a pandas series.
        :param supplied_mapping: mapping as list of dicts. They actual mapping can be either a dict or pd.Series
        :return: the mapping with all actual mappings being pandas series
        """
        msg = "Invalid supplied mapping, must be of type List[Dict[str, Union[Dict, pd.Series]]]." \
              "For an example refer to the documentation"
        if not isinstance(supplied_mapping, list):
            raise ValueError(msg)
        for mapping_el in supplied_mapping:
            if not isinstance(mapping_el, dict):
                raise ValueError(msg)
            if "col" not in mapping_el:
                raise KeyError("Mapping must contain a key 'col' for each column to encode")
            if "mapping" not in mapping_el:
                raise KeyError("Mapping must contain a key 'mapping' for each column to encode")
            mapping = mapping_el["mapping"]
            if isinstance(mapping_el, dict):
                # convert to dict in order to standardise
                mapping_el["mapping"] = pd.Series(mapping)
            if "data_type" not in mapping_el:
                mapping_el["data_type"] = mapping_el["mapping"].index.dtype
        return supplied_mapping

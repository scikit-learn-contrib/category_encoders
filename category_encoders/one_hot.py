"""One-hot or dummy coding"""
import numpy as np
import pandas as pd
import warnings
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'willmcginnis'


class OneHotEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):
    """Onehot (or dummy) coding for categorical features, produces one feature per category, each binary.

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
    use_cat_names: bool
        if True, category values will be included in the encoded column names. Since this can result in duplicate column names, duplicates are suffixed with '#' symbol until a unique name is generated.
        If False, category indices will be used instead of the category values.
    handle_unknown: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.

        'error' will raise a `ValueError` at transform time if there are new categories.
        'return_nan' will encode a new value as `np.nan` in every dummy column.
        'value' will encode a new value as 0 in every dummy column.
        'indicator' will add an additional dummy column (in both training and test data).
    handle_missing: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.

        'error' will raise a `ValueError` if missings are encountered.
        'return_nan' will encode a missing value as `np.nan` in every dummy column.
        'value' will encode a missing value as 0 in every dummy column.
        'indicator' will treat missingness as its own category, adding an additional dummy column
        (whether there are missing values in the training set or not).
        'ignore' will encode missing values as 0 in every dummy column, NOT adding an additional category.
        

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name="house_prices", as_frame=True)
    >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = OneHotEncoder(cols=['CentralAir', 'Heating'], handle_unknown='indicator').fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 15 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   float64
     1   MSSubClass     1460 non-null   float64
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   YearBuilt      1460 non-null   float64
     5   Heating_1      1460 non-null   int64  
     6   Heating_2      1460 non-null   int64  
     7   Heating_3      1460 non-null   int64  
     8   Heating_4      1460 non-null   int64  
     9   Heating_5      1460 non-null   int64  
     10  Heating_6      1460 non-null   int64  
     11  Heating_-1     1460 non-null   int64  
     12  CentralAir_1   1460 non-null   int64  
     13  CentralAir_2   1460 non-null   int64  
     14  CentralAir_-1  1460 non-null   int64  
    dtypes: float64(4), int64(10), object(1)
    memory usage: 171.2+ KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://ibgwww.colorado.edu/~carey/p5741ndir/Coding_Categorical_Variables.pdf
    
    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_N_UNIQUE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_missing='value', handle_unknown='value', use_cat_names=False):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.mapping = None
        self.ordinal_encoder = None
        self.use_cat_names = use_cat_names

    @property
    def category_mapping(self):
        return self.mapping

    def _fit(self, X, y=None, **kwargs):
        oe_missing_strat = {
            'error': 'error',
            'return_nan': 'return_nan',
            'value': 'value',
            'indicator': 'return_nan',
            'ignore': 'return_nan'
        }[self.handle_missing]
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing=oe_missing_strat,
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        self.mapping = self.generate_mapping()

    def generate_mapping(self):
        mapping = []
        found_column_counts = {}

        for switch in self.ordinal_encoder.mapping:
            col = switch.get('col')
            values = switch.get('mapping').copy(deep=True)

            if self.handle_missing == 'value':
                values = values[values > 0]

            if len(values) == 0:
                continue

            index = []
            new_columns = []

            append_nan_to_index = False
            for cat_name, class_ in values.items():
                if pd.isna(cat_name) and self.handle_missing in ['return_nan', 'ignore']:
                    # we don't want a mapping column if return_nan
                    # but do add the index to the end
                    append_nan_to_index = class_
                    continue
                if self.use_cat_names:
                    n_col_name = f"{col}_{cat_name}"
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                else:
                    n_col_name = f"{col}_{class_}"

                index.append(class_)
                new_columns.append(n_col_name)

            if self.handle_unknown == 'indicator':
                n_col_name = f"{col}_-1"
                if self.use_cat_names:
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                new_columns.append(n_col_name)
                index.append(-1)

            if append_nan_to_index:
                index.append(append_nan_to_index)

            base_matrix = np.eye(N=len(index), M=len(new_columns), dtype=int)
            base_df = pd.DataFrame(data=base_matrix, columns=new_columns, index=index)

            if self.handle_unknown == 'value':
                base_df.loc[-1] = 0
            elif self.handle_unknown == 'return_nan':
                base_df.loc[-1] = np.nan

            if self.handle_missing == 'return_nan':
                base_df.loc[-2] = np.nan
            elif self.handle_missing in ['value','ignore']:
                base_df.loc[-2] = 0

            mapping.append({'col': col, 'mapping': base_df})

        return mapping

    def _transform(self, X):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.get_dummies(X)
        return X

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data.

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
        X = util.convert_input(X_in, columns=self.feature_names_out_, deep=True)

        X = self.reverse_dummies(X, self.mapping)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError(f"Unexpected input dimension {X.shape[1]}, the attribute drop_invariant should "
                                 "be False when transforming the data")
            else:
                raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self._dim}')

        if not list(self.cols):
            return X if self.return_df else X.to_numpy()

        for switch in self.ordinal_encoder.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

            if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
                for col in self.cols:
                    if X[switch.get('col')].isna().any():
                        warnings.warn("inverse_transform is not supported because transform impute "
                                      f"the unknown category nan when encode {col}")

        return X if self.return_df else X.to_numpy()

    def get_dummies(self, X_in):
        """
        Convert numerical variable into dummy variables

        Parameters
        ----------
        X_in: DataFrame

        Returns
        -------
        dummies : DataFrame

        """

        X = X_in.copy(deep=True)

        cols = X.columns.tolist()

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col].fillna(-2))
            base_df = base_df.set_index(X.index)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        X = X.reindex(columns=cols)

        return X

    def reverse_dummies(self, X, mapping):
        """
        Convert dummy variable into numerical variables

        Parameters
        ----------
        X : DataFrame
        mapping: list-like
              Contains mappings of column to be transformed to it's new columns and value represented

        Returns
        -------
        numerical: DataFrame

        """
        out_cols = X.columns.tolist()
        mapped_columns = []
        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            insert_at = out_cols.index(mod.columns[0])

            X.insert(insert_at, col, 0)
            positive_indexes = mod.index[mod.index > 0]
            for i in range(positive_indexes.shape[0]):
                existing_col = mod.columns[i]
                val = positive_indexes[i]
                X.loc[X[existing_col] == 1, col] = val
                mapped_columns.append(existing_col)
            X = X.drop(mod.columns, axis=1)
            out_cols = X.columns.tolist()

        return X

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
import category_encoders.utils as util


class RankHotEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):
    """The rank-hot encoder is similar to a one-hot encoder,
    except every feature up to and including the current rank is hot.
    This is also called thermometer encoding.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    use_cat_names: bool
        if True, category values will be included in the encoded column names.
           Since this can result in duplicate column names,
           duplicates are suffixed with '#' symbol until a unique name is generated.
        If False, category indices will be used instead of the category values.
    handle_unknown: str
        options are 'error', 'value', 'return_nan'.
        The default is 'value'.
        'value': If an unknown label occurrs, it is represented as 0 array.
        'error': If an unknown label occurrs, error message is displayed.
        'return_nan': If an unknown label occurrs, np.nan is returned in all columns.
    handle_missing: str
        options are 'error', 'value' and 'return_nan'. The default is 'value'.
        Missing value also considered as unknown value in the final data set.

    Example
    -------
     >>> from category_encoders import *
     >>> import pandas as pd
     >>> from sklearn.datasets import fetch_openml
     >>> bunch = fetch_openml(name="house_prices", as_frame=True)
     >>> display_cols = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "YearBuilt", "Heating", "CentralAir"]
     >>> y = bunch.target
     >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
     >>> enc = RankHotEncoder(cols=['CentralAir', 'Heating'], handle_unknown='indicator').fit(X, y)
     >>> numeric_dataset = enc.transform(X)
     >>> print(numeric_dataset.info())
     <class 'pandas.core.frame.DataFrame'>
     RangeIndex: 1460 entries, 0 to 1459
     Data columns (total 13 columns):
      #   Column        Non-Null Count  Dtype  
     ---  ------        --------------  -----  
      0   Id            1460 non-null   float64
      1   MSSubClass    1460 non-null   float64
      2   MSZoning      1460 non-null   object 
      3   LotFrontage   1201 non-null   float64
      4   YearBuilt     1460 non-null   float64
      5   Heating_1     1460 non-null   int64  
      6   Heating_2     1460 non-null   int64  
      7   Heating_3     1460 non-null   int64  
      8   Heating_4     1460 non-null   int64  
      9   Heating_5     1460 non-null   int64  
      10  Heating_6     1460 non-null   int64  
      11  CentralAir_1  1460 non-null   int64  
      12  CentralAir_2  1460 non-null   int64  
     dtypes: float64(4), int64(8), object(1)
     memory usage: 148.4+ KB
     None
    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_N_UNIQUE

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_missing="value",
        handle_unknown="value",
        use_cat_names=None,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self._dim = None
        self.mapping = None
        self.use_cat_names = use_cat_names

    def _fit(self, X, y, **kwargs):
        oe_missing_strat = {
            'error': 'error',
            'return_nan': 'return_nan',
            'value': 'value',
            'indicator': 'return_nan',
        }[self.handle_missing]
        # supply custom mapping in order to assure order of ordinal variable
        ordered_mapping = []
        for col in self.cols:
            oe_col = OrdinalEncoder(verbose=self.verbose, cols=[col], handle_unknown="value", handle_missing=oe_missing_strat)
            oe_col.fit(X[col].sort_values().to_frame(name=col))
            ordered_mapping += oe_col.mapping

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown="value", handle_missing=oe_missing_strat, mapping=ordered_mapping
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        self.mapping = self.generate_mapping()

        return self

    def _transform(self, X_in, override_return_df=False):
        X = X_in.copy(deep=True)
        X = self.ordinal_encoder.transform(X)
        input_cols = X.columns.tolist()

        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError("Columns to be encoded can not contain new values")

        for switch, ordinal_switch in zip(self.mapping, self.ordinal_encoder.category_mapping):
            col = switch.get("col")
            mod = switch.get("mapping")
            encode_feature_series = X[col]

            unknow_elements = encode_feature_series[encode_feature_series == -1]

            encoding_dict = {i: list(row.values()) for i, row in mod.to_dict(orient="index").items()}
            if self.handle_unknown == "value":
                default_value = [0] * len(encoding_dict)
            elif self.handle_unknown == "return_nan":
                default_value = [np.nan] * len(encoding_dict)
            elif self.handle_unknown == "error":
                if not unknow_elements.empty:
                    unknowns_str = ', '.join([str(x) for x in unknow_elements.unique()])
                    msg = f"Unseen values {unknowns_str} during transform in column {col}."
                    raise ValueError(msg)
                default_value = [0] * len(encoding_dict)
            else:
                raise ValueError(f"invalid option for 'handle_unknown' parameter: {self.handle_unknown}")

            def apply_coding(row: pd.Series):
                val = row.iloc[0]
                if pd.isna(val):
                    if self.handle_missing == "value":
                        return default_value
                    elif self.handle_missing == "return_nan":
                        return [np.nan] * len(default_value)
                    else:
                        raise ValueError("Unhandled nan")
                return encoding_dict.get(row.iloc[0], default_value)

            encoded = encode_feature_series.to_frame().apply(apply_coding, axis=1, result_type="expand")
            encoded.columns = mod.columns

            X = pd.concat([encoded, X], axis=1)

            old_column_index = input_cols.index(col)
            input_cols[old_column_index:old_column_index + 1] = mod.columns
        X = X.reindex(columns=input_cols)

        return X

    def create_dataframe(self, X, encoded, key_col):

        if not (isinstance(encoded, pd.DataFrame) or isinstance(encoded, pd.Series)):
            encoded = pd.DataFrame(encoded, columns=key_col)

        X_ = pd.concat([encoded, X], axis=1)
        return X_

    def inverse_transform(self, X_in):
        X = X_in.copy(deep=True)
        cols = X.columns.tolist()
        if self._dim is None:
            raise ValueError("Must train encoder before it can be used to inverse_transform data")

        for switch, ordinal_mapping in zip(self.mapping, self.ordinal_encoder.category_mapping):
            col = switch.get("col")
            cats = switch.get("mapping")
            if col != ordinal_mapping.get("col"):
                raise ValueError("Column order of OrdinalEncoder and RankHotEncoder do not match")
            inv_map = {v: k for k, v in ordinal_mapping.get("mapping").to_dict().items()}

            arrs = X[cats.columns]
            reencode = arrs.sum(axis=1).rename(col)

            orig_dtype = ordinal_mapping.get("data_type")
            reencode2 = reencode.replace(inv_map).astype(orig_dtype)
            if np.any(reencode2[:] == 0):
                reencode2[reencode2[:] == 0] = np.nan

            X = self.create_dataframe(X, reencode2, col)

            first_inex = cols.index(cats.columns[0])
            last_index = cols.index(cats.columns[-1]) + 1

            del cols[first_inex:last_index]
            cols.insert(self.ordinal_encoder.feature_names_out_.index(col), col)

        X = X.reindex(columns=cols)

        return X

    def generate_mapping(self):
        mapping = []
        found_column_counts = {}

        for switch in self.ordinal_encoder.mapping:
            col: str = switch.get("col")
            values: pd.Series = switch.get("mapping").copy(deep=True)

            if self.handle_missing == "value":
                values = values[values > 0]

            if len(values) == 0:
                continue

            index = []
            new_columns = []

            for cat_name, class_ in values.items():
                if self.use_cat_names:
                    n_col_name = f"{col}_{cat_name}"
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += "#" * found_count
                else:
                    n_col_name = f"{col}_{class_}"

                index.append(class_)
                new_columns.append(n_col_name)

            base_matrix = np.tril(np.ones((len(index), len(index)), dtype=int))
            base_df = pd.DataFrame(data=base_matrix, columns=new_columns, index=index)

            mapping.append({"col": col, "mapping": base_df})
        return mapping

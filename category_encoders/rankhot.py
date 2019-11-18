import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util

class RankHotEncoder(BaseEstimator, TransformerMixin):
    """The rank-hot encoder is similar to a one-hot encoder, except every feature up to and including the current rank is hot.

       Parameters
       ----------

       verbose: int
           integer indicating verbosity of the output. 0 for none.
       cols: list
           a list of columns to encode, if None, all string columns will be encoded.
       drop_invariant: bool
           boolean for whether or not to drop columns with 0 variance.
       use_cat_names: bool
           if True, category values will be included in the encoded column names. Since this can result in duplicate column names, duplicates are suffixed with '#' symbol until a unique name is generated.
           If False, category indices will be used instead of the category values.
       handle_unknown: str
           options are 'error', 'ignore'. The default is 'ignore'. If unknown variable exists then it is represented as 0 array. Else error message
           is displayed
       handle_missing: str
           options are 'error' and 'value'. The default is 'value'. Missing value also considered as unknown value in the final data set.

       Example
       -------
       >>> from category_encoders import *
       >>> import pandas as pd
       >>> from sklearn.datasets import load_boston
       >>> bunch = load_boston()
       >>> y = bunch.target
       >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
       >>> enc = RankHotEncoder(cols=['CHAS'], handle_unknown='ignore').fit(X)
       >>> numeric_dataset = enc.transform(X)
       >>> print(numeric_dataset.info())
    """

    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS_1     506 non-null int32
    CHAS_2     506 non-null int32
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64   
    """
    def __init__(self, verbose = 0, cols = None, drop_invariant = False, handle_missing='value', handle_unknown='ignore', use_cat_names = None):
        self.cols = cols
        self.ordinal_encoder = None
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self._dim = None
        self.verbose = verbose
        self.mapping = None
        self.feature_names = None
        self.drop_invariant = drop_invariant
        self.use_cat_names = use_cat_names

    def fit(self, X):

        self._dim = X.shape[0]
        self.feature_names = X.columns

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )

        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        self.mapping = self.generate_mapping()

        self.ordinal_encoder.category_mapping[0]['mapping'] =  self.ordinal_encoder.category_mapping[0]['mapping'].drop(labels = [float('nan')])

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = list(X_temp.columns)

        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))

        return self

    def transform(self, X_in, override_return_df=False):
        X = X_in.copy(deep=True)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        if self._dim is None:
            raise ValueError(
                'Must train encoder before it can be used to transform data.')

        X = util.convert_input(X)
        cols = X.columns.values.tolist()

        # then make sure that it is the right size
        if X.shape[0] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (
                X.shape[0], self._dim,))

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            key_value = list(filter(lambda person: person['col'] == col, self.ordinal_encoder.category_mapping))[0].get('mapping').to_dict()
            encoded_key_value = {k: v for k, v in key_value.items() if pd.notna(k)}
            encode_feature_series = X[col]

            unknown_indicator = False

            unknow_elements = encode_feature_series[~encode_feature_series.isin(encoded_key_value)]

            if not unknow_elements.empty:
                if self.handle_unknown == 'ignore':
                    if encode_feature_series.isnull().values.any():

                        try:
                            encode_feature_series.fillna(value=0, inplace=True)
                        except ValueError:
                            if encode_feature_series.dtype == 'category':
                                encode_feature_series.cat.add_categories('Unknown', inplace=True)
                                encode_feature_series.fillna(value='Unknown', inplace=True)
                                encoded_key_value.update({'Unknown': 0})
                    else:
                        for un in unknow_elements:
                            encode_feature_series[encode_feature_series == un] = 0
                    unknown_indicator = True
                    if 'Unknown' not in encoded_key_value.keys():
                        encoded_key_value.update({0: 0})

                if self.handle_unknown == 'error':
                    print("Unknown value appear in the Data set")

            encoded = np.vectorize(encoded_key_value.get)(encode_feature_series)

            if unknown_indicator:
                if 'Unknown' in encoded_key_value.keys():
                    del encoded_key_value['Unknown']
                else:
                    del encoded_key_value[0]

            encoding = (np.arange(mod.shape[0]) < np.array(encoded).reshape(-1, 1)).astype(int)

            X = self.create_dataframe(X, encoding, mod.columns)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns
        X = X.reindex(columns=cols)

        return X

    def encode(self, X, mappings):
        try:
            encoded = np.array([mappings[v] for v in X])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return encoded

    def create_dataframe(self, X, encoded, key_col):

        if not (isinstance(encoded, pd.DataFrame) or isinstance(encoded, pd.Series)):
            encoded = pd.DataFrame(encoded, columns=key_col)

        X_ = pd.concat([encoded, X], axis=1)
        return X_

    def inverse_transform(self, X_in):
        X = X_in.copy(deep=True)
        cols = X.columns.values.tolist()
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        for sqitch in self.mapping:
            col = sqitch.get('col')
            cats = sqitch.get('mapping')
            encodedList = self.ordinal_encoder.category_mapping[:]['col' == col].get('mapping').to_dict()

            arrs = X[cats.columns]
            bool_array = arrs.astype(bool)
            reencode = bool_array.sum(axis=1).rename(col)

            inv_map = {v: k for k, v in encodedList.items()}
            reencode2 = reencode.replace(inv_map)
            if np.any(reencode2[:] == 0):
                reencode2[reencode2[:] == 0] = 'None'

            X = self.create_dataframe(X,reencode2, col)

            first_inex = cols.index(cats.columns[0])
            last_index = cols.index(cats.columns[-1]) + 1

            del cols[first_inex:last_index]
            cols.insert(self.ordinal_encoder.feature_names.index(col), col)

        X = X.reindex(columns=cols)

        return X

    def return_maps(self, arr, col):
        return self.ordinal_encoder.category_mapping[:]['col' == col].get('mapping').index.values[arr-1]

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

            for cat_name, class_ in values.iteritems():
                if self.use_cat_names:
                    n_col_name = str(col) + '_%s' % (cat_name,)
                    found_count = found_column_counts.get(n_col_name, 0)
                    found_column_counts[n_col_name] = found_count + 1
                    n_col_name += '#' * found_count
                else:
                    n_col_name = str(col) + '_%s' % (class_,)

                index.append(class_)
                new_columns.append(n_col_name)

            base_matrix = np.eye(N=len(index), dtype=np.int)
            base_df = pd.DataFrame(data=base_matrix, columns=new_columns, index=index)

            mapping.append({'col': col, 'mapping': base_df})
        return mapping

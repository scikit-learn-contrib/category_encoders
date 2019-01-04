import pandas as pd
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np

import category_encoders.tests.test_utils as tu

import category_encoders as encoders

mapping = {"A": "A|B", "B": "A|B", "C": "C", "D": "D"}


def mask_X_extra(df, mapping, col="extra"):
    df = df.copy()
    shuffle_idx = np.arange(df.shape[0]) % 2 == 0
    df[col + "_mask"] = df[col]
    df.loc[shuffle_idx, col + "_mask"] = df.loc[shuffle_idx, col + "_mask"].map(mapping)
    nan_idx = np.arange(df.shape[0]) % 9 == 0
    df[col + "_mask_withnan"] = df[col + "_mask"]
    df.loc[nan_idx, col + "_mask_withnan"] = np.nan
    return df


np_X = tu.create_array(n_rows=100)
np_X_t = tu.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = tu.create_dataset(n_rows=100)
X_t = tu.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)
X_mask = X.pipe(mask_X_extra, mapping)
X_t_mask = X_t.pipe(mask_X_extra, mapping)


def get_first_extract_column(df):
    first_extract_column = df.columns.str.extract("(.*)_[1-9]+", expand=True).dropna()
    first_extract_column = set(first_extract_column.values.reshape(first_extract_column.shape[0])).pop()
    return first_extract_column


class TestMultiHotEncoderTestCase(TestCase):

    def test_multi_hot_fit(self):
        enc = encoders.MultiHotEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertEqual(enc.transform(X_t).shape[1],
                         enc.transform(X_t[X_t['extra'] != 'A']).shape[1],
                         'We have to get the same count of columns')

        self.assertEqual(enc.transform(X_t).shape[1],
                         enc.transform(X_t[X_t['extra'] != 'D']).shape[1],
                         'We have to get the same count of columns')

        enc = encoders.MultiHotEncoder(verbose=1, return_df=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len([x for x in out.columns.values if str(x).startswith('extra_')]), 3, "We have three columns whose names start with extra_")

        enc = encoders.MultiHotEncoder(verbose=1, return_df=True, cols=["categorical"])
        enc.fit(X)
        out = enc.transform(X_t)
        first_extract_column = get_first_extract_column(out)
        self.assertEqual(first_extract_column, "categorical", "We have to transform only categorical column")

    def test_multi_hot_AmbiguousCase(self):
        enc = encoders.MultiHotEncoder(verbose=1, return_df=False)
        enc.fit(X_mask)
        self.assertEqual(enc.transform(X_t_mask).shape[1],
                         enc.transform(X_t_mask[X_t_mask['extra'] != 'A']).shape[1],
                         'We have to get the same count of columns')

        enc = encoders.MultiHotEncoder(verbose=1, return_df=True)
        enc.fit(X_mask)
        out = enc.transform(X_t_mask)
        extra_column_length = out.columns.str.extract("(extra_*)_[1-9]+", expand=True).dropna().shape[0]
        self.assertEqual(extra_column_length, 3, "We have to contain 3 extra columns")

        extra_mask_columns = out.columns.str.extract("(extra_.*)_[1-9]+", expand=True).dropna()
        set_extra_mask_columns = set(extra_mask_columns.values.reshape(extra_mask_columns.shape[0]))
        self.assertSetEqual(set_extra_mask_columns, set(['extra_mask', 'extra_mask_withnan']), "We have to contain extra_mask_* and extra_mask_withnan_* columns")

        extra_mask_column_length = out.columns.str.extract("(extra_mask_*)_[1-9]+", expand=True).dropna().shape[0]
        self.assertEqual(extra_mask_column_length, 3, "We have to contain 3 extra_mask columns")

        extra_mask_withnan_column_length = out.columns.str.extract("(extra_mask_.*)_[1-9]+", expand=True).dropna().shape[0]
        self.assertEqual(extra_mask_withnan_column_length, 4, "We have to contain 4 extra_mask_withnan columns")

        enc = encoders.MultiHotEncoder(verbose=1, return_df=True, cols=["extra_mask"])
        enc.fit(X_mask)
        out = enc.transform(X_t_mask)
        first_extract_column = get_first_extract_column(out)
        self.assertEqual(first_extract_column, "extra_mask", "We have to transform only extra_mask column")

    def test_multi_hot_OutputAnalysis(self):
        enc = encoders.MultiHotEncoder(verbose=1, return_df=True, cols=["extra_mask"])
        enc.fit(X_mask)
        out = enc.transform(X_t_mask)
        extra_transformed_columns = out.columns[out.columns.str.match(".+_[1-9]+").fillna(False).values.astype(bool)]

        extra_D_index = X_t_mask["extra"] == "D"
        extra_D_transformed_sum = (out.loc[extra_D_index, extra_transformed_columns] != 0).sum().sum()
        self.assertEqual(extra_D_transformed_sum, 0, "Every input whose extra value is D has to be transformed into [0,0,0]")

        extra_notD_transformed_sum = ((out.loc[~extra_D_index, extra_transformed_columns] != 0).sum(axis=1) <= 0).sum()
        self.assertEqual(extra_notD_transformed_sum, 0, "Every input whose extra value is not D has to be transformed into vector except [0,0,0]")

        ambiguous_index = X_t_mask.extra_mask.map(lambda x: "|" in x)
        ambiguous_transformed_sum = ((out.loc[ambiguous_index, extra_transformed_columns] > 0).sum(axis=1) != 2).sum()
        self.assertEqual(ambiguous_transformed_sum, 0, "Every input whose extra_mask contains ambigous input (A|B) has to be transformed into [a,b,c] where a > 0 and b > 0 and c = 0")

    def test_multi_hot_fit_transform(self):
        enc = encoders.MultiHotEncoder(cols=["extra"])
        out = enc.fit_transform(X_mask)
        first_extract_column = get_first_extract_column(out)
        self.assertEqual(first_extract_column, "extra", "We have to transform only categorical column")

        self.assertEqual(enc.transform(X_t_mask).shape[1],
                         enc.transform(X_t_mask[X_t_mask['extra'] != 'A']).shape[1],
                         'We have to get the same count of columns')

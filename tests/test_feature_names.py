import numpy as np
import pandas as pd
import tests.helpers as th
from numpy.testing import assert_array_equal
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as encoders
from unittest import TestCase

__author__ = 'JaimeArboleda'

# data definitions
X = th.create_dataset(n_rows=100)
cat_columns = ['categorical', 'na_categorical']
num_columns = ['float']
X = X[cat_columns + num_columns]
np_y = np.random.randn(X.shape[0]) > 0.5
y = pd.DataFrame(np_y)

class TestEncodersFeaturesOut(TestCase):

    def test_feature_names_out(self):
        if sklearn.__version__ < "1.2.0":
            return
        else:
            sklearn.set_config(transform_output="pandas")
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)()
                X_t = encoder.fit_transform(X, y)

                categorical_preprocessor_start = Pipeline(
                    steps=[
                        ("encoder", getattr(encoders, encoder_name)())
                    ]
                )
                categorical_preprocessor_middle = Pipeline(
                    steps=[
                        ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
                        ("encoder", getattr(encoders, encoder_name)())
                    ]
                )
                numerical_preprocessor = Pipeline(
                    steps=[
                        ("imputation_constant", SimpleImputer(fill_value=0, strategy="constant"))
                    ]
                )
                preprocessor = ColumnTransformer(
                    [
                        ("categorical_prep_start", categorical_preprocessor_start, ["categorical", "na_categorical"]),
                        ("categorical_prep_middle", categorical_preprocessor_middle, ["categorical", "na_categorical"]),
                        ("numerical_prep", numerical_preprocessor, ["float"])
                    ]
                )
                X_tt = preprocessor.fit_transform(X, y)

                assert_array_equal(
                    np.array(X_t.columns),
                    encoder.get_feature_names_out()
                )
                assert_array_equal(
                    np.array(X_tt.columns),
                    preprocessor.get_feature_names_out()
                )
                assert_array_equal(
                    np.array(
                        [
                            c
                            for c in X_t.columns
                            if c not in num_columns
                        ]
                    ),
                    np.array(
                        [
                            c[len("categorical_prep_start__"):]
                            for c in X_tt.columns
                            if "categorical_prep_start" in c
                        ]
                    )
                )
                assert_array_equal(
                    np.array(
                        [
                            c
                            for c in X_t.columns
                            if c not in num_columns
                        ]
                    ),
                    np.array(
                        [
                            c[len("categorical_prep_middle__"):]
                            for c in X_tt.columns
                            if "categorical_prep_middle" in c
                        ]
                    )
                )

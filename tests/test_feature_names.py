import numpy as np
import pandas as pd
import tests.helpers as th
from numpy.testing import assert_array_equal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as encoders

__author__ = 'JaimeArboleda'

# data definitions
X = th.create_dataset(n_rows=100)
np_y = np.random.randn(X.shape[0]) > 0.5
y = pd.DataFrame(np_y)
feature_names = np.array(X.columns)


def test_feature_names_out():
    target_encoder = encoders.TargetEncoder()
    target_encoder.fit(X=X, y=y)
    assert_array_equal(feature_names, target_encoder.get_feature_names_out(input_features=None))


def test_feature_names_out_in_sklearn_ensemble():
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessor_target = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("target", encoders.TargetEncoder(handle_unknown="value")),
        ]
    )

    categorical_preprocessor_ohe = Pipeline(
        steps=[
            ("imputation_constant", SimpleImputer(fill_value="missing", strategy="constant")),
            ("ohe", encoders.OneHotEncoder(handle_unknown="value", use_cat_names=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("categorical_target", categorical_preprocessor_target, ["categorical", "na_categorical"]),
            ("categorical_ohe", categorical_preprocessor_ohe, ["categorical", "na_categorical"]),
            ("numerical", numeric_preprocessor, ["float"])
        ]
    )
    preprocessor.fit(X, y)
    assert_array_equal(
        preprocessor.get_feature_names_out(),
        np.array([
            "categorical_target__categorical",
            "categorical_target__na_categorical",
            "categorical_ohe__categorical_C",
            "categorical_ohe__categorical_B",
            "categorical_ohe__categorical_A",
            "categorical_ohe__na_categorical_A",
            "categorical_ohe__na_categorical_B",
            "categorical_ohe__na_categorical_C",
            "categorical_ohe__na_categorical_missing",
            "numerical__float"
        ])
    )



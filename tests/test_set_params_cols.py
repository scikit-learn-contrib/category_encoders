"""Column-selection regression tests for sklearn parameter updates."""

import numpy as np
import pandas as pd
import pytest
from category_encoders import OneHotEncoder, OrdinalEncoder, SummaryEncoder
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline


@pytest.mark.parametrize('encoder_cls', [OrdinalEncoder, SummaryEncoder])
@pytest.mark.parametrize(
    'initial, selected', [(None, ['a']), (['a'], None), ('all', ['a']), (['a'], 'ALL'), (None, [])]
)
def test_set_params_columns_before_and_after_fit(encoder_cls, initial, selected):
    """Explicit column changes must match construction, including after fitting."""
    X = pd.DataFrame({'a': ['x', 'y'] * 4, 'b': ['p', 'q'] * 4, 'n': [1, 2] * 4})
    y = pd.Series([0.0, 1.0, 2.0, 3.0] * 2)
    expected = encoder_cls(cols=selected).fit_transform(X, y)
    for fitted in [False, True]:
        encoder = encoder_cls(cols=initial)
        if fitted:
            encoder.fit(X, y)
        assert encoder.set_params(cols=selected) is encoder
        assert_frame_equal(encoder.fit_transform(X, y), expected)


@pytest.mark.parametrize('encoder_cls', [OrdinalEncoder, SummaryEncoder])
@pytest.mark.parametrize('cols', [None, 'all'])
def test_unrelated_parameter_keeps_automatic_refit(encoder_cls, cols):
    """Unrelated updates must not turn automatic selection into explicit columns."""
    X = pd.DataFrame({'a': ['x', 'y'] * 4, 'n': [1, 2] * 4})
    y = pd.Series([0.0, 1.0, 2.0, 3.0] * 2)
    encoder = encoder_cls(cols=cols).fit(X, y)
    renamed = X.rename(columns={'a': 'b', 'n': 'm'})
    encoder.set_params(verbose=0)
    assert_frame_equal(
        encoder.fit_transform(renamed, y), encoder_cls(cols=cols).fit_transform(renamed, y)
    )


@pytest.mark.parametrize('encoder_cls', [OrdinalEncoder, SummaryEncoder])
def test_set_params_numpy_column_selection(encoder_cls):
    """Array inputs must use newly requested positional columns."""
    X = np.array([['x', 'p'], ['y', 'q']] * 4)
    y = np.array([0.0, 1.0, 2.0, 3.0] * 2)
    encoder = encoder_cls().set_params(cols=[0])
    assert_frame_equal(encoder.fit_transform(X, y), encoder_cls(cols=[0]).fit_transform(X, y))


def test_gridsearch_pipeline_column_selection():
    """GridSearchCV must fit the requested encoding configurations in a pipeline."""
    X = pd.DataFrame({'a': [0, 1, 2, 0, 1, 2] * 4, 'b': [0, 0, 1, 1, 2, 2] * 4})
    y = pd.Series([0.0, 2.0, 1.0, 1.0, 3.0, 2.0] * 4)
    pipeline = Pipeline([('encode', OneHotEncoder()), ('regress', Ridge())])
    search = GridSearchCV(
        pipeline, {'encode__cols': [['a'], ['b']]}, cv=3, scoring='neg_mean_squared_error'
    ).fit(X, y)
    expected = [
        cross_val_score(
            Pipeline([('encode', OneHotEncoder(cols=[col])), ('regress', Ridge())]),
            X,
            y,
            cv=3,
            scoring='neg_mean_squared_error',
        ).mean()
        for col in ['a', 'b']
    ]
    np.testing.assert_allclose(search.cv_results_['mean_test_score'], expected)
    assert search.best_estimator_.named_steps['encode'].cols == search.best_params_['encode__cols']


@pytest.mark.parametrize('encoder_cls', [OrdinalEncoder, SummaryEncoder])
def test_set_params_sklearn_contract(encoder_cls):
    """The override must preserve validation and unrelated parameter updates."""
    encoder = encoder_cls()
    assert encoder.set_params() is encoder
    assert encoder.set_params(return_df=False) is encoder
    assert encoder.get_params()['return_df'] is False
    with pytest.raises(ValueError, match='Invalid parameter'):
        encoder.set_params(nonexistent_parameter=True)

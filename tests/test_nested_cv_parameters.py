"""Behavior of fitted nested-CV parameters in estimator searches."""

import numpy as np
import pandas as pd
import pytest
from category_encoders import LeaveOneOutEncoder
from category_encoders.wrapper import NestedCVWrapper
from numpy.testing import assert_allclose
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline


@pytest.mark.parametrize(
    'updates', [{'cv': 2}, {'random_state': 5}, {'shuffle': False, 'random_state': None}]
)
def test_nested_cv_uses_updated_split_parameters(updates):
    """Parameter updates control the actual out-of-fold training partitions."""
    rng = np.random.RandomState(8)
    X = pd.DataFrame({'cat': rng.choice(['a', 'b', 'c'], 36)})
    y = pd.Series(rng.randint(0, 2, 36))
    wrapper = NestedCVWrapper(LeaveOneOutEncoder(), cv=3, random_state=7)
    wrapper.set_params(**updates)
    actual = wrapper.fit_transform(X, y)['cat'].to_numpy()

    params = {'n_splits': 3, 'shuffle': True, 'random_state': 7}
    params.update({('n_splits' if key == 'cv' else key): val for key, val in updates.items()})
    expected = np.empty(len(X))
    for train, valid in StratifiedKFold(**params).split(X, y):
        stats = y.iloc[train].groupby(X.iloc[train]['cat']).agg(['mean', 'count'])
        prior = y.iloc[train].mean()
        means = stats['mean'].where(stats['count'] > 1, prior)
        expected[valid] = X.iloc[valid]['cat'].map(means).fillna(prior)
    assert_allclose(actual, expected)


def test_nested_cv_integer_grid_search():
    """A pipeline can tune the documented integer cv parameter."""
    rng = np.random.RandomState(8)
    X = pd.DataFrame({'cat': rng.choice(['a', 'b', 'c'], 36)})
    y = rng.randint(0, 2, 36)
    pipeline = Pipeline(
        [
            ('encoder', NestedCVWrapper(LeaveOneOutEncoder(), random_state=7)),
            ('model', LogisticRegression()),
        ]
    )
    search = GridSearchCV(pipeline, {'encoder__cv': [2, 3]}, cv=3, error_score='raise')
    search.fit(X, y)
    assert np.isfinite(search.cv_results_['mean_test_score']).all()


def test_nested_cv_preserves_custom_splitter():
    """A caller-supplied splitter ignores the wrapper's shuffle parameters."""
    X = pd.DataFrame({'cat': ['a', 'b', 'a', 'b', 'a', 'b']})
    y = pd.Series([0, 1, 1, 0, 0, 1])
    cv = KFold(3)
    wrapper = NestedCVWrapper(LeaveOneOutEncoder(), cv=cv)
    expected = wrapper.fit_transform(X, y)
    actual = wrapper.set_params(shuffle=True, random_state=7).fit_transform(X, y)
    assert wrapper.cv is cv
    assert_allclose(actual, expected)

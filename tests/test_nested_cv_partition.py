"""Validate the test-fold partition required for out-of-fold encoding."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from category_encoders import LeaveOneOutEncoder
from category_encoders.wrapper import NestedCVWrapper
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    PredefinedSplit,
    RepeatedKFold,
    ShuffleSplit,
    TimeSeriesSplit,
)


class FixedCV:
    """Yield fixed test positions, training on their complement."""

    def __init__(self, test_folds):
        """Store the requested test folds."""
        self.test_folds = test_folds

    def split(self, X, y=None, groups=None):
        """Yield each test fold once."""
        for test in self.test_folds:
            test = np.asarray(test)
            yield np.setdiff1d(np.arange(len(X)), test), test


@pytest.fixture
def data():
    """Return categorical features, a passthrough column, and binary targets."""
    X = pd.DataFrame({'cat': list('aabb') * 3, 'row': np.arange(12)})
    y = pd.Series([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    return X, y


@pytest.mark.parametrize(
    'cv',
    [
        TimeSeriesSplit(n_splits=3),
        RepeatedKFold(n_splits=3, n_repeats=2, random_state=0),
        ShuffleSplit(n_splits=3, test_size=0.25, random_state=0),
        PredefinedSplit([-1] * 3 + [0] * 3 + [1] * 3 + [2] * 3),
        FixedCV([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 10]]),
        FixedCV([[0, 1, 2, 3], [4, 5, 6, 7], [7, 8, 9, 10]]),
        FixedCV([]),
    ],
    ids=[
        'time-series',
        'repeated',
        'shuffle',
        'predefined-missing',
        'within-fold',
        'overlap',
        'empty',
    ],
)
def test_non_partition_cv_is_rejected(data, cv):
    """Reject missing or repeated test rows instead of returning corrupt encodings."""
    X, y = data
    wrapper = NestedCVWrapper(LeaveOneOutEncoder(), cv=cv)

    with pytest.raises(ValueError, match='test folds must include each row exactly once'):
        wrapper.fit_transform(X, y)


def test_invalid_cv_is_rejected_before_fitting(data):
    """Validate coverage before fitting any fold encoder or the full-data encoder."""
    X, y = data
    wrapper = NestedCVWrapper(LeaveOneOutEncoder(), cv=TimeSeriesSplit(n_splits=3))

    with patch.object(LeaveOneOutEncoder, 'fit') as fit:
        with pytest.raises(ValueError, match='test folds must include each row exactly once'):
            wrapper.fit_transform(X, y)
        fit.assert_not_called()


@pytest.mark.parametrize(
    'cv',
    [
        KFold(n_splits=3, shuffle=True, random_state=0),
        GroupKFold(n_splits=3),
        PredefinedSplit([0, 1, 2] * 4),
        FixedCV([[11, 9, 7, 5], [3, 1, 10, 8], [6, 4, 2, 0]]),
    ],
    ids=['kfold', 'grouped', 'predefined', 'unsorted'],
)
@pytest.mark.parametrize('duplicate_index', [False, True])
def test_partition_encodings_match_fold_means(data, cv, duplicate_index):
    """Keep fold-local means, input order, and held-out inference for valid partitions."""
    X, y = data
    X.index = y.index = np.arange(len(X)) % 3 if duplicate_index else np.arange(50, 62)
    groups = np.repeat(np.arange(3), 4) if isinstance(cv, GroupKFold) else None
    original = X.copy(deep=True)
    original_y = y.copy(deep=True)
    expected = X.copy()
    expected['cat'] = np.nan

    for train, test in cv.split(X, y, groups):
        train_labels = y.iloc[train].to_numpy()
        train_categories = X.iloc[train]['cat'].to_numpy()
        means = pd.Series(train_labels).groupby(train_categories).mean()
        expected.iloc[test, 0] = X.iloc[test]['cat'].map(means).to_numpy()

    wrapper = NestedCVWrapper(LeaveOneOutEncoder(cols=['cat']), cv=cv)
    result, first_test, second_test = wrapper.fit_transform(X, y, X_test=(X, X), groups=groups)

    pd.testing.assert_frame_equal(result, expected)
    full_data = LeaveOneOutEncoder(cols=['cat']).fit(X, y).transform(X)
    pd.testing.assert_frame_equal(first_test, full_data)
    pd.testing.assert_frame_equal(second_test, full_data)
    pd.testing.assert_frame_equal(wrapper.transform(X), full_data)
    pd.testing.assert_frame_equal(X, original)
    pd.testing.assert_series_equal(y, original_y)

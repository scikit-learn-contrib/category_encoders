"""Index alignment regressions for the multiclass target-encoding wrapper."""

import numpy as np
import pandas as pd
import pytest
from category_encoders import LeaveOneOutEncoder, TargetEncoder
from category_encoders.wrapper import PolynomialWrapper


@pytest.mark.parametrize('encoder_cls', [TargetEncoder, LeaveOneOutEncoder])
@pytest.mark.parametrize('target_type', ['list', 'array', 'series'])
@pytest.mark.parametrize('index_values', [range(6), [15, 10, 13, 11, 14, 12], [2, 2, 1, 1, 0, 0]])
def test_fit_transform_preserves_input_index(encoder_cls, target_type, index_values):
    """Array-like targets inherit the feature index before label encoding."""
    X = pd.DataFrame(
        {'cat': ['a', 'b', 'a', 'b', 'a', 'b'], 'numeric': np.arange(6)},
        index=pd.Index(index_values, name='sample'),
    )
    original = X.copy(deep=True)
    labels = ['first', 'second', 'third', 'first', 'second', 'third']
    if target_type == 'array':
        y = np.array(labels)
    elif target_type == 'series':
        y = pd.Series(labels, index=X.index, name='outcome')
    else:
        y = labels

    reference = PolynomialWrapper(encoder_cls(cols=['cat']))
    expected = reference.fit_transform(X.reset_index(drop=True), labels)
    wrapper = PolynomialWrapper(encoder_cls(cols=['cat']))
    result = wrapper.fit_transform(X, y)

    pd.testing.assert_index_equal(result.index, X.index)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
    pd.testing.assert_series_equal(result['numeric'], X['numeric'])
    pd.testing.assert_frame_equal(X, original)
    pd.testing.assert_frame_equal(wrapper.transform(X, y), result)
    pd.testing.assert_frame_equal(
        wrapper.transform(X).reset_index(drop=True),
        reference.transform(X.reset_index(drop=True)),
    )


@pytest.mark.parametrize('encoder_cls', [TargetEncoder, LeaveOneOutEncoder])
def test_array_features_use_series_target_index(encoder_cls):
    """A pandas target supplies the index when features are a NumPy array."""
    X = np.array([['a'], ['b'], ['a'], ['b'], ['a'], ['b']])
    y = pd.Series(
        ['first', 'second', 'third', 'first', 'second', 'third'],
        index=pd.Index([15, 10, 13, 11, 14, 12], name='sample'),
    )
    expected = PolynomialWrapper(encoder_cls()).fit_transform(X, y.to_numpy())
    result = PolynomialWrapper(encoder_cls()).fit_transform(X, y)

    pd.testing.assert_index_equal(result.index, y.index)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


@pytest.mark.parametrize('encoder_cls', [TargetEncoder, LeaveOneOutEncoder])
def test_mismatched_pandas_indexes_still_raise(encoder_cls):
    """Do not silently reindex pandas targets that disagree with the features."""
    X = pd.DataFrame({'cat': ['a', 'b', 'a', 'b', 'a', 'b']})
    y = pd.Series(['first', 'second', 'third'] * 2, index=np.arange(10, 16))
    wrapper = PolynomialWrapper(encoder_cls())
    wrapper.fit(X, y.to_numpy())

    with pytest.raises(ValueError, match='indexes, but they do not match'):
        wrapper.transform(X, y)

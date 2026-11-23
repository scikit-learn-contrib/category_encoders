"""Regression tests for the sklearn wrapper fit return contract."""

import numpy as np
import pandas as pd
import pytest
from category_encoders import TargetEncoder
from category_encoders.wrapper import NestedCVWrapper, PolynomialWrapper


@pytest.mark.parametrize('wrapper_class', [NestedCVWrapper, PolynomialWrapper])
@pytest.mark.parametrize('as_array', [False, True])
@pytest.mark.parametrize('y', [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]])
def test_wrapper_fit_returns_fitted_estimator(wrapper_class, as_array, y):
    """Fit returns the same trained wrapper and supports chained inference."""
    X = pd.DataFrame({'category': ['a', 'b', 'a', 'b', 'a', 'b'], 'numeric': range(6)})
    X_test = pd.DataFrame({'category': ['a', 'b', 'unseen'], 'numeric': [10, 11, 12]})
    if as_array:
        X, X_test = X.to_numpy(), X_test.to_numpy()
    control = wrapper_class(TargetEncoder(cols=[0] if as_array else ['category']))
    control.fit(X, y)
    expected = control.transform(X_test)

    wrapper = wrapper_class(TargetEncoder(cols=[0] if as_array else ['category']))
    assert wrapper.fit(X, y) is wrapper
    pd.testing.assert_frame_equal(wrapper.transform(X_test), expected)
    pd.testing.assert_frame_equal(wrapper.fit(X, y).transform(X_test), expected)

    # Refitting returns this instance too, with encodings for the new target.
    new_y = np.roll(y, 1)
    control.fit(X, new_y)
    pd.testing.assert_frame_equal(
        wrapper.fit(X, new_y).transform(X_test), control.transform(X_test)
    )

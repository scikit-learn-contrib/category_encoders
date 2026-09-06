"""Tests for the clone contract: fittedness checks and error messages (Issue #232)."""

from unittest import TestCase

import category_encoders as ce
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


def make_data():
    """Small deterministic dataset: one categorical feature, one binary target."""
    return pd.DataFrame(
        {'A': ['Hello', 'World!', 'ML', 'is', 'interesting'] * 10, 'B': [1, 0, 0, 1, 1] * 10}
    )


class ToyClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that predicts with a pre-fitted encoder; reproduces Issue #232."""

    def __init__(self, encoder=None):
        self.encoder = encoder
        self.reg = SVC(C=1000)

    def fit(self, X, y):
        """Refit the wrapped classifier on the features encoded by the pre-fitted encoder."""
        X = self.encoder.transform(X)
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        """Predict with the features encoded by the pre-fitted encoder."""
        X = self.encoder.transform(X)
        return self.reg.predict(X)


class PreFittedClassifier(ToyClassifier):
    """ToyClassifier that preserves the pre-fitted encoder across clones (sklearn >= 1.6)."""

    def __sklearn_clone__(self):
        """Preserve this instance, including the pre-fitted encoder."""
        return self


class TestCloneContract(TestCase):
    """A cloned or never-fitted encoder must raise an informative NotFittedError."""

    def test_unfitted_encoder_transform_raises_notfitted(self):
        """A never-fitted encoder raises NotFittedError, not a lower-level error."""
        X = make_data()[['A']]
        for encoder_name in ('BinaryEncoder', 'OneHotEncoder', 'OrdinalEncoder'):
            with self.subTest(encoder_name=encoder_name):
                with pytest.raises(NotFittedError, match='not fitted'):
                    getattr(ce, encoder_name)().transform(X)

    def test_fittedness_checked_before_handle_missing_scan(self):
        """Fittedness is checked before the handle_missing='error' null scan (Issue #232)."""
        X = make_data()[['A']]
        with pytest.raises(NotFittedError, match='not fitted'):
            ce.BinaryEncoder(handle_missing='error').transform(X)

    def test_clone_of_fitted_encoder_is_unfitted(self):
        """A clone of a fitted encoder is unfitted: transform raises, check_is_fitted agrees."""
        X = make_data()[['A']]
        cloned = clone(ce.BinaryEncoder().fit(X))
        with pytest.raises(NotFittedError, match='clone'):
            cloned.transform(X)
        with pytest.raises(NotFittedError):
            check_is_fitted(cloned)

    def test_cross_val_predict_with_prefitted_encoder_raises_informative_error(self):
        """Issue #232: cross-validating a wrapper around a pre-fitted encoder explains the clone."""
        data = make_data()
        encoder = ce.BinaryEncoder().fit(data[['A']])
        with pytest.raises(NotFittedError) as excinfo:
            cross_val_predict(ToyClassifier(encoder=encoder), data[['A']], data['B'], cv=2)
        message = str(excinfo.value)
        assert 'not fitted' in message
        assert 'clone' in message
        assert 'Must train encoder before it can be used to transform data' not in message

    def test_prefitted_holder_pattern_allows_cross_val_predict(self):
        """The __sklearn_clone__ holder pattern works with cross_val_predict."""
        data = make_data()
        encoder = ce.BinaryEncoder().fit(data[['A']])
        predictions = cross_val_predict(
            PreFittedClassifier(encoder=encoder), data[['A']], data['B'], cv=2
        )
        assert len(predictions) == len(data)

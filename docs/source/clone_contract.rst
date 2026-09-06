Cloning and cross-validation
============================

All encoders follow the scikit-learn estimator contract, including ``sklearn.base.clone``:
cloning re-creates an estimator from its constructor parameters, and a clone is **always
unfitted**, even when the original encoder was fitted.

Why clones are unfitted
-----------------------

``clone()`` must return an unfitted estimator. That guarantee is what makes
cross-validation, grid searches, and pipelines safe: every fold or candidate receives its
own fresh estimator, so no information can leak between folds through fitted state.
Fitted attributes such as ``n_features_in_`` or ``feature_names_in_`` are deliberately not
carried over.

The cross-validation pitfall
----------------------------

Passing a *fitted* encoder as a constructor parameter of another estimator does not
survive cloning. When scikit-learn clones the outer estimator (for example inside
``cross_val_predict``), it also clones the encoder, and that clone is unfitted:

.. code-block:: python

    import pandas as pd
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.model_selection import cross_val_predict
    from sklearn.svm import SVC
    import category_encoders as ce

    data = pd.DataFrame({'A': ['a', 'b', 'c'] * 20, 'B': [0, 1, 1] * 20})
    encoder = ce.BinaryEncoder().fit(data[['A']])

    class ToyClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, encoder=None):
            self.encoder = encoder
            self.reg = SVC(C=1000)

        def fit(self, X, y):
            X = self.encoder.transform(X)
            self.reg.fit(X, y)
            return self

        def predict(self, X):
            X = self.encoder.transform(X)
            return self.reg.predict(X)

    # raises NotFittedError: within each fold the encoder is an unfitted clone
    cross_val_predict(ToyClassifier(encoder=encoder), data[['A']], data['B'])

Transforming with an unfitted encoder raises ``NotFittedError`` with a message that names
the problem and the ways out.

Recommended patterns
--------------------

**Fit the encoder inside the pipeline or the cross-validation loop** (recommended). Put
the encoder in a ``Pipeline`` and cross-validate the pipeline; the encoder is refit on
each training fold, which is also the statistically correct treatment:

.. code-block:: python

    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(ce.BinaryEncoder(), SVC(C=1000))
    cross_val_predict(pipe, data[['A']], data['B'])

**Hold a pre-fitted encoder with ``__sklearn_clone__``** (scikit-learn >= 1.6). If you
genuinely want one global fit shared across all folds — for example because fitting is
expensive — give the *wrapper* a ``__sklearn_clone__`` method that preserves the encoder:

.. code-block:: python

    class PreFittedClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, encoder=None):
            self.encoder = encoder
            self.reg = SVC(C=1000)

        def fit(self, X, y):
            X = self.encoder.transform(X)
            self.reg.fit(X, y)
            return self

        def predict(self, X):
            X = self.encoder.transform(X)
            return self.reg.predict(X)

        def __sklearn_clone__(self):
            return self  # keep the pre-fitted encoder; the SVC is refit per fold

    cross_val_predict(PreFittedClassifier(encoder=encoder), data[['A']], data['B'])

Note that this wrapper deliberately steps outside the usual scikit-learn isolation rules:
cloning no longer produces a fresh, independent copy, so use it only when a shared global
fit is really what you want.

Why the encoders themselves do not implement ``__sklearn_clone__``
------------------------------------------------------------------

Implementing ``__sklearn_clone__`` on the encoders to return the fitted instance would
break the clone-is-unfitted guarantee for every other consumer: the clones that
cross-validation, grid search, and pipelines create would silently share fitted state and
leak information between folds. If you need a shared fit, make that explicit in your own
wrapper, as shown above.

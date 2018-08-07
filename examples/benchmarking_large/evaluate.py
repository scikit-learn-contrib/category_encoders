import time

import numpy as np
import sklearn
import sklearn.metrics.scorer
from memory_profiler import memory_usage
from pympler import asizeof
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, Imputer

from examples.benchmarking_large import customizedCV

class Columns(BaseEstimator, TransformerMixin):
    """
    A transformer, which selects only some of the columns based on their names.
    Will be replaced with columnTransformers:
    	https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
    from scikit-learn 20.0 once it will be released.
    Warning: We assume that the input is a DataFrame.
    """

    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.names]

class EncoderWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper, which measures encoder's memory consumption.
    """

    def __init__(self, encoder=None):
        self.encoder = encoder
        self.original_df_mem = None     # Size of the original DataFrame
        self.encoded_df_mem = None      # Size of the encoded DataFrame
        self.blank_encoder_mem = None   # Size of the blank encoder
        self.trained_encoder_mem = None # Size of the trained encoder
        self.fit_peak_mem = None        # Peak memory consumption in MiB during fitting
        self.score_peak_mem = None      # Peak memory consumption in MiB during scoring
        self.fit_encoder_time = None
        self.score_encoder_time = None

    def fit(self, X, y=None, **fit_params):
        self.original_df_mem = X.memory_usage(deep=True).sum()
        self.blank_encoder_mem = asizeof.asizeof(self.encoder)
        start_time = time.time()
        self.fit_peak_mem = memory_usage(proc=(self.encoder.fit, (X, y)), max_usage=True)[0]
        self.fit_encoder_time = time.time() - start_time
        self.trained_encoder_mem = asizeof.asizeof(self.encoder)
        return self

    def transform(self, X):
        start_time = time.time()
        score_peak_mem, out = memory_usage(proc=(self.encoder.transform, (X,)), retval=True, max_usage=True)
        self.score_encoder_time = time.time() - start_time
        self.score_peak_mem = score_peak_mem[0]
        self.encoded_df_mem = out.memory_usage(deep=True).sum()
        return out

def evaluate(X, y, fold_count, encoder, model, class_values):
    numeric = list(X.select_dtypes(include=[np.number]).columns.values)
    categorical = list(X.select_dtypes(include='object').columns.values)

    if len(numeric) == 0:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ('categorical', make_pipeline(Columns(names=categorical), EncoderWrapper(encoder)))
            ])),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ('numeric', make_pipeline(Columns(names=numeric), Imputer(), StandardScaler())),
                ('categorical', make_pipeline(Columns(names=categorical), EncoderWrapper(encoder)))
            ])),
            ('model', model)
        ])

    # Make a dictionary of the scorers.
    # Justification of the choices:
    #   Matthews correlation coefficient: represents thresholding measures
    #   AUC: represents ranking measures
    #   Brier score: represents calibration measures
    # Beware:
    #   All measures that accept "labels" parameter are passed a list with all the unique class values, otherwise they may complain during the cross-validation.
    #   F-measure is sensitive to the definition of the positive class. In our case, positive class is the majority class.
    log_loss_score = sklearn.metrics.make_scorer(sklearn.metrics.log_loss, needs_proba=True, labels=class_values)
    accuracy_score = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
    f1_macro = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, labels=class_values)
    brier = sklearn.metrics.make_scorer(sklearn.metrics.brier_score_loss)
    auc = sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)
    matthews = sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef)
    kappa = sklearn.metrics.make_scorer(sklearn.metrics.cohen_kappa_score, labels=class_values)
    score_dict = {'accuracy': accuracy_score, 'f1_macro': f1_macro, 'log_loss': log_loss_score, 'auc':auc, 'brier':brier, 'matthews':matthews, 'kappa':kappa}

    # Perform cross-validation.
    # It returns a dict containing training scores, fit-times and score-times in addition to the test score.
    # Beware: We monkey-patch the CV in order to measure the memory utilization.
    sklearn.model_selection._validation._fit_and_score = customizedCV._fit_and_score
    scores = customizedCV.cross_validate(estimator=pipeline, X=X, y=y, scoring=score_dict, cv=StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=2001), return_train_score=True)

    return scores

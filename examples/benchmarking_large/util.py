import time
import warnings
from copy import deepcopy

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def train_encoder(X, y, fold_count, encoder):
    """
    Defines folds and performs the data preprocessing (categorical encoding, NaN imputation, normalization)
    Returns a list with {X_train, y_train, X_test, y_test}, average fit_encoder_time and average score_encoder_time

    Note: We normalize all features (not only numerical features) because otherwise SVM would
        get stuck for hours on ordinal encoded cylinder.bands.arff dataset due to presence of
        unproportionally high values.

    Note: The fold count is variable because there are datasets, which have less than 10 samples in the minority class.

    Note: We do not use pipelines because of:
        https://github.com/scikit-learn/scikit-learn/issues/11832
    """
    kf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=2001)
    encoder = deepcopy(encoder)  # Because of https://github.com/scikit-learn-contrib/categorical-encoding/issues/106
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    folds = []
    fit_encoder_time = 0
    score_encoder_time = 0

    for train_index, test_index in kf.split(X, y):
        # Split data
        X_train, X_test = X.iloc[train_index, :].reset_index(drop=True), X.iloc[test_index, :].reset_index(drop=True)
        y_train, y_test = y[train_index].reset_index(drop=True), y[test_index].reset_index(drop=True)

        # Training
        start_time = time.time()
        X_train = encoder.fit_transform(X_train, y_train)
        fit_encoder_time += time.time() - start_time
        X_train = imputer.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train)

        # Testing
        start_time = time.time()
        X_test = encoder.transform(X_test)
        score_encoder_time += time.time() - start_time
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)

        folds.append([X_train, y_train, X_test, y_test])

    return folds, fit_encoder_time/fold_count, score_encoder_time/fold_count

def train_model(folds, model):
    """
    Evaluation with:
      Matthews correlation coefficient: represents thresholding measures
      AUC: represents ranking measures
      Brier score: represents calibration measures
    """
    scores = []
    fit_model_time = 0      # Sum of all the time spend on fitting the training data, later on normalized
    score_model_time = 0    # Sum of all the time spend on scoring the testing data, later on normalized

    for X_train, y_train, X_test, y_test in folds:
        # Training
        start_time = time.time()
        with ignore_warnings(category=ConvergenceWarning):  # Yes, neural networks do not always converge
            model.fit(X_train, y_train)
        fit_model_time += time.time() - start_time
        prediction_train_proba = model.predict_proba(X_train)[:, 1]
        prediction_train = (prediction_train_proba >= 0.5).astype('uint8')

        # Testing
        start_time = time.time()
        prediction_test_proba = model.predict_proba(X_test)[:, 1]
        score_model_time += time.time() - start_time
        prediction_test = (prediction_test_proba >= 0.5).astype('uint8')

        # When all the predictions are of a single class, we get a RuntimeWarning in matthews_corr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores.append([
                sklearn.metrics.matthews_corrcoef(y_test, prediction_test),
                sklearn.metrics.matthews_corrcoef(y_train, prediction_train),
                sklearn.metrics.roc_auc_score(y_test, prediction_test_proba),
                sklearn.metrics.roc_auc_score(y_train, prediction_train_proba),
                sklearn.metrics.brier_score_loss(y_test, prediction_test_proba),
                sklearn.metrics.brier_score_loss(y_train, prediction_train_proba)
            ])

    return np.mean(scores, axis=0), fit_model_time/len(folds), score_model_time/len(folds)

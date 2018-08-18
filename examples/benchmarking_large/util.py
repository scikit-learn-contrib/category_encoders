import time
from copy import deepcopy

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, Imputer


def train_encoder(X, y, fold_count, encoder):
    """
    Defines folds and performs the data preprocessing
    Returns a list with {X_train, y_train, X_test, y_test}, average fit_encoder_time and average score_encoder_time

    Note: We do not use pipelines because of:
        https://github.com/scikit-learn/scikit-learn/issues/11832
    """
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2001)
    encoder = deepcopy(encoder) # Because of https://github.com/scikit-learn-contrib/categorical-encoding/issues/106
    folds = []
    fit_encoder_time = 0
    score_encoder_time = 0

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index, :].reset_index(drop=True), X.iloc[test_index, :].reset_index(drop=True)
        y_train, y_test = y[train_index].reset_index(drop=True), y[test_index].reset_index(drop=True)

        # Fill NaN and normalize
        X_train, X_test = _process_numerical_features(X_train, X_test)

        # Training
        start_time = time.time()
        X_train = encoder.fit_transform(X_train, y_train)
        fit_encoder_time += time.time() - start_time

        # Testing
        start_time = time.time()
        X_test = encoder.transform(X_test)
        score_encoder_time += time.time() - start_time

        folds.append([X_train, y_train, X_test, y_test])

    return folds, fit_encoder_time/fold_count, score_encoder_time/fold_count

def _process_numerical_features(X_train, X_test):
    """
    Impute missing values and normalize numerical features with z-score normalization
    """
    imputer = Imputer(strategy='mean', axis=0)
    scaler = StandardScaler()
    numeric = list(X_train.select_dtypes(include=[np.number]).columns.values)
    non_numeric = list(X_train.select_dtypes(exclude=[np.number]).columns.values)

    if len(numeric) > 0:
        # Training
        X_train_num = imputer.fit_transform(X_train[numeric])
        X_train_num = scaler.fit_transform(X_train_num)
        X_train = np.hstack([X_train_num, X_train[non_numeric]])

        # Testing
        X_test_num = imputer.transform(X_test[numeric])
        X_test_num = scaler.transform(X_test_num)
        X_test = np.hstack([X_test_num, X_test[non_numeric]])

    return [X_train, X_test]

def train_model(folds, model):
    """
    Evaluation with:
      Matthews correlation coefficient: represents thresholding measures
      AUC: represents ranking measures
      Brier score: represents calibration measures
    """
    scores = []

    for X_train, y_train, X_test, y_test in folds:
        # Training
        model.fit(X_train, y_train)
        prediction_train_proba = model.predict_proba(X_train)[:, 1]
        prediction_train = (prediction_train_proba >= 0.5).astype('uint8')

        # Testing
        prediction_test_proba = model.predict_proba(X_test)[:, 1]
        prediction_test = (prediction_test_proba >= 0.5).astype('uint8')

        scores.append([
            sklearn.metrics.matthews_corrcoef(y_test, prediction_test),
            sklearn.metrics.matthews_corrcoef(y_train, prediction_train),
            sklearn.metrics.roc_auc_score(y_test, prediction_test_proba),
            sklearn.metrics.roc_auc_score(y_train, prediction_train_proba),
            sklearn.metrics.brier_score_loss(y_test, prediction_test_proba),
            sklearn.metrics.brier_score_loss(y_train, prediction_train_proba)
        ])

    return np.mean(scores, axis=0)
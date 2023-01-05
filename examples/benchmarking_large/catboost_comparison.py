"""
Compare performance of CatBoost internal categorical encoding with our categorical encoding.
Conclusion: CatBoost beats our encoders by large margin.
"""
import os

import pandas as pd
import numpy as np
from catboost import Pool, cv, CatBoostClassifier

import category_encoders
from examples.benchmarking_large import arff_loader, csv_loader
from examples.benchmarking_large.util import train_model, train_encoder

# The settings are taken from:
#   Data-driven advice for applying machine learning to bioinformatics problems, Olson et al.
model = CatBoostClassifier(iterations=50, max_depth=3)

# We use Arff datasets on GitHub. But once OpenML loader will be part of scikit-learn:
#   https://github.com/scikit-learn/scikit-learn/pull/11419
# the plan is to move on OpenML.
# We ignore datasets without any polynomial feature.
# We also ignore 'splice.arff', 'anneal.arff', 'anneal.orig.arff' due to high runtime.
# Datasets sensitive to amount of regularization are:
#     audiology.arff                    Medium impact   (contains missing values)
#     breast.cancer.arff                Medium impact
#     bridges.version1.arff             Medium impact   (contains an ID + missing values)
#     bridges.version2.arff                             (contains an ID)
#     car.arff
#     colic.arff
#     cylinder.bands.arff               Large impact    (contains a constant column + almost an ID)
#     flags.arff                        Large impact
#     heart.c.arff                                      (contains missing values)
#     hepatitis.arff
#     hypothyroid.arff                                  (contains a constant column)
#     kr.vs.kp.arff
#     labor.arff                        Large impact
#     lymph.arff
#     nursery.arff
#     postoperative.patient.data.arff   Large impact    (testing AUC is commonly <0.5, see: https://www.openml.org/t/4528)
#     primary.tumor.arff                                (contains missing values)
#     solar.flare1.arff                 Medium impact
#     solar.flare2.arff                 Medium impact
#     soybean.arff                      Large impact
#     sick.arff
#     spectrometer.arff                 Medium impact   (contains an ID)
#     sponge.arff                       Large impact
#     tic-tac-toe.arff
#     trains.arff                       Medium impact   (tiny dataset -> with high variance)
datasets = [#'audiology.arff',
            'autos.arff', 'breast.cancer.arff', 'bridges.version1.arff', 'bridges.version2.arff', 'car.arff',
            'colic.arff', 'credit.a.arff', 'credit.g.arff', 'cylinder.bands.arff', 'flags.arff', 'heart.c.arff', 'heart.h.arff',
            'hepatitis.arff', 'hypothyroid.arff', 'kr.vs.kp.arff', 'labor.arff', 'lymph.arff', 'mushroom.arff', 'nursery.arff',
            'postoperative.patient.data.arff', 'primary.tumor.arff', 'sick.arff', 'solar.flare1.arff', 'solar.flare2.arff',
            'soybean.arff', 'spectrometer.arff', 'sponge.arff', 'tic-tac-toe.arff', 'trains.arff', 'vote.arff', 'vowel.arff']

datasets = ['carvana.csv', 'erasmus.csv', 'internetusage.csv', 'ipumsla97small.csv', 'kobe.csv', 'pbcseq.csv', 'phpvcoG8S.csv', 'westnile.csv'] # amazon is too large...


# We painstakingly initialize each encoder here because that gives us the freedom to initialize the
# encoders with any setting we want.
encoders = [ #category_encoders.BackwardDifferenceEncoder(),
             category_encoders.BaseNEncoder(),
             category_encoders.BinaryEncoder(),
             category_encoders.HashingEncoder(),
             # category_encoders.HelmertEncoder(),
             category_encoders.JamesSteinEncoder(),
             category_encoders.LeaveOneOutEncoder(),
             category_encoders.MEstimateEncoder(),
             category_encoders.OneHotEncoder(),
             category_encoders.OrdinalEncoder(),
             # category_encoders.PolynomialEncoder(),
             # category_encoders.SumEncoder(),
             category_encoders.TargetEncoder(),
             category_encoders.WOEEncoder()]

encoders = [category_encoders.TargetEncoder(), category_encoders.JamesSteinEncoder(), category_encoders.WOEEncoder()]

# Initialization
if os.path.isfile('./output/result.csv'):
    os.remove('./output/result.csv')

# Loop over datasets, then over encoders
for dataset_name in datasets:
    # X, y, fold_count = arff_loader.load(dataset_name)
    X, y, fold_count, nominal_columns = csv_loader.load(dataset_name)

    # Get indexes (not names) of categorical features
    categorical_indexes = []
    for col in X.select_dtypes(exclude=[np.number]).columns.values:
        for i, col2 in enumerate(X.columns):
            if col == col2:
                categorical_indexes.append(i)

    # Simple missing value treatment
    X.fillna(-999, inplace=True)

    # Perform cross-validation
    pool = Pool(X, y, categorical_indexes)
    params = {'iterations': 50,
              'depth': 3,
              'loss_function': 'Logloss',
              'eval_metric': 'AUC',
              'verbose': False}
    scores = cv(pool, params, logging_level='Silent')
    auc = scores.iloc[-1,0]


    # Log into csv
    result = pd.DataFrame([dataset_name, y.name, 'CatBoost', 'default', model.__class__.__name__, X.shape[1],
                           '', '', '', '', '', '', '', auc, '', '', '']).T
    if not os.path.isfile('./output/result.csv'):
        result.to_csv('./output/result.csv',
                      header=['dataset', 'target', 'encoder', 'encoder_setting', 'model', 'input_features', 'output_features',
                              'fit_encoder_time', 'score_encoder_time', 'fit_model_time', 'score_model_time', 'test_matthews',
                              'train_matthews', 'test_auc', 'train_auc', 'test_brier', 'train_brier'], index=False)
    else:
        result.to_csv('./output/result.csv', mode='a', header=False, index=False)

    # Our encoding
    for encoder in encoders:
        print("Encoding:", dataset_name, y.name, encoder.__class__.__name__)
        folds, fit_encoder_time, score_encoder_time = train_encoder(X, y, fold_count, encoder)

        print('Evaluating:', dataset_name, encoder.__class__.__name__, model.__class__.__name__)
        scores, fit_model_time, score_model_time = train_model(folds, model)

        # Log into csv
        result = pd.DataFrame([dataset_name, y.name, encoder.__class__.__name__, encoder.__dict__, model.__class__.__name__, X.shape[1],
                               folds[0][0].shape[1], fit_encoder_time, score_encoder_time, fit_model_time, score_model_time]
                              + list(scores)).T
        if not os.path.isfile('./output/result.csv'):
            result.to_csv('./output/result.csv',
                          header=['dataset', 'target', 'encoder', 'encoder_setting', 'model', 'input_features', 'output_features',
                                  'fit_encoder_time', 'score_encoder_time', 'fit_model_time', 'score_model_time', 'test_matthews',
                                  'train_matthews', 'test_auc', 'train_auc', 'test_brier', 'train_brier'], index=False)
        else:
            result.to_csv('./output/result.csv', mode='a', header=False, index=False)

print('Finished. The result was stored into ./output/result.csv.')
print(result)

"""
Run a large scale benchmark.

We measure: {dataset, encoder, model, train and test accuracy measures, train and test runtimes, feature count}.
"""
import os
import warnings

import numpy as np
import pandas as pd
import category_encoders
from examples.benchmarking_large import arff_loader
from examples.benchmarking_large.evaluate import evaluate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Setting
models = [SGDClassifier(loss='modified_huber', max_iter=50, tol=1e-3),
          LogisticRegression(),
          SVC(kernel='linear', probability=True),
          KNeighborsClassifier(),
          GaussianNB(),
          DecisionTreeClassifier(),
          GradientBoostingClassifier(),
          RandomForestClassifier(),
          MLPClassifier()]

# We use Arff datasets on GitHub. But once OpenML loader will be part of scikit-learn:
#   https://github.com/scikit-learn/scikit-learn/pull/11419
# the plan is to move on OpenML.
# We ignore 'breast.w.arff' and 'zoo.arff' because they use integer data type, which is not supported by the used arff library.
# We also ignore 'splice.arff', 'anneal.arff', 'anneal.orig.arff' due to high runtime.
datasets = ['wine.arff', 'arrhythmia.arff', 'audiology.arff', 'autos.arff', 'balance.scale.arff',
            'balance.scale.arff', 'breast.cancer.arff', 'bridges.version1.arff', 'bridges.version2.arff', 'car.arff', 'cmc.arff',
            'colic.arff', 'colic.orig.arff', 'column2C.arff', 'column3C.arff', 'credit.a.arff', 'credit.g.arff', 'cylinder.bands.arff',
            'dermatology.arff', 'diabetes.arff', 'ecoli.arff', 'flags.arff', 'glass.arff', 'haberman.arff', 'heart.c.arff', 'heart.h.arff',
            'heart.statlog.arff', 'hepatitis.arff', 'hypothyroid.arff', 'ionosphere.arff', 'iris.arff', 'kr.vs.kp.arff', 'labor.arff',
            'letter.arff', 'lymph.arff', 'mushroom.arff', 'nursery.arff', 'optdigits.arff', 'page.blocks.arff', 'pendigits.arff',
            'postoperative.patient.data.arff', 'primary.tumor.arff', 'segment.arff', 'shuttle.landing.control.arff', 'sick.arff',
            'solar.flare1.arff', 'solar.flare2.arff', 'sonar.arff', 'soybean.arff', 'spambase.arff', 'spect.test.arff', 'spect.train.arff',
            'spectf.test.arff', 'spectf.train.arff', 'spectrometer.arff', 'sponge.arff', 'tae.arff', 'tic-tac-toe.arff', 'trains.arff',
            'vehicle.arff', 'vote.arff', 'vowel.arff', 'waveform5000.arff']

# We ignore encoders {BackwardDifferenceEncoder, HelmertEncoder, PolynomialEncoder and SumEncoder} because of:
#   https://github.com/scikit-learn-contrib/categorical-encoding/issues/91
encoders = [category_encoders.BaseNEncoder(), category_encoders.OneHotEncoder(), category_encoders.BinaryEncoder(),
            category_encoders.HashingEncoder(), category_encoders.OrdinalEncoder(), category_encoders.TargetEncoder(),
            category_encoders.LeaveOneOutEncoder()]

# Initialization
if os.path.isfile('./output/result.csv'):
    os.remove('./output/result.csv')

# Ok...
warnings.filterwarnings('ignore')

# Loop over datasets, then over encoders, and finally, over the models
for dataset_name in datasets:
    X, y, unique_y, fold_count = arff_loader.load(dataset_name)
    for encoder in encoders:
        for model in models:
            print('Running:', dataset_name, y.name, encoder.__class__.__name__, model.__class__.__name__)
            scores = evaluate(X, y, fold_count, encoder, model, unique_y)

            # Calculate averages over folds
            aggregates = list(np.mean(scores[key]) for key in scores)

            # Log into csv to have something in case of a crash
            result = pd.DataFrame([dataset_name, y.name, encoder.__class__.__name__, model.__class__.__name__, X.shape[1]] + aggregates).T
            if not os.path.isfile('./output/result.csv'):  # If the file does not exist, add header
                result.to_csv('./output/result.csv', header=['dataset', 'target', 'encoder', 'model', 'features'] + list(scores), index=False)
            else:  # Else it exists, so append without writing the header
                result.to_csv('./output/result.csv', mode='a', header=False, index=False)

print('Finished. A csv file was stored into ./output directory.')

"""
Run a large scale benchmark on data from "penn-ml-benchmarks".

We measure: {dataset, encoder, model, fold, train and test accuracy measures, fit and score runtimes, fit and score memory consumption, feature count}.

Note: To decrease runtime, we cache datasets, and trained encoders into "datasets" and "encoders" directories respectively.

Threats to validity:

1) The decision what is a positive class and what is a negative class is left up to the implementation (by default pos=1).
Consequently, measures like F-measure, which depend on the definition of a positive class, will provide misleading
results if the positive class is something else than 1.

2) Integer attributes are always treated as numerical even though they are actually nominal.

3) In order to get reproducible results, we use a fixed random seed.

4) We use default settings for models and encoders.

"""
import os
import warnings

import numpy as np
import pandas as pd
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

import category_encoders

# Setting
# datasets = pmlb.classification_dataset_names    # try all classification datasets from pmlb
# encoders = category_encoders.__all__            # try each encoding method available
models = [LogisticRegression(),
          SGDClassifier(loss='modified_huber', max_iter=50, tol=1e-3),
          SVC(kernel='linear', probability=True),
          KNeighborsClassifier(),
          GaussianNB(),
          BernoulliNB(),
          DecisionTreeClassifier(),
          GradientBoostingClassifier(),
          RandomForestClassifier(),
          MLPClassifier(),
          ElasticNet()
          ]  # classifiers

# For debugging
# datasets = ['anneal.arff','breast.cancer.arff', 'bridges.version1.arff', 'autos.arff' , 'balance.scale.arff', 'bridges.version2.arff', 'car.arff', 'cmc.arff', 'colic.arff', 'colic.orig.arff', 'column2C.arff', 'column3C.arff', 'credit.a.arff', 'credit.g.arff', 'cylinder.bands.arff', 'dermatology.arff', 'diabetes.arff']
datasets = ['ecoli.arff', 'glass.arff', 'haberman.arff', 'heart.c.arff', 'heart.h.arff', 'heart.statlog.arff', 'hepatitis.arff', 'hypothyroid.arff', 'ionosphere.arff', 'iris.arff', 'kr.vs.kp.arff', 'labor.arff', 'letter.arff', 'lymph.arff', 'mushroom.arff', 'nursery.arff', 'optdigits.arff', 'page.blocks.arff', 'pendigits.arff']
# datasets = [ 'segment.arff', 'shuttle.landing.control.arff', 'sick.arff', 'solar.flare1.arff', 'solar.flare2.arff', 'sonar.arff', 'soybean.arff', 'spambase.arff', 'spect.test.arff', 'spect.train.arff', 'spectf.test.arff', 'spectf.train.arff']
# datasets = [ 'sponge.arff', 'tae.arff', 'tic-tac-toe.arff', 'trains.arff', 'vehicle.arff', 'vote.arff', 'vowel.arff', 'waveform5000.arff', 'wine.arff', 'anneal.arff', 'anneal.orig.arff', 'arrhythmia.arff']
# datasets = ['audiology.arff','flags.arff', 'postoperative.patient.data.arff', 'primary.tumor.arff', 'spectrometer.arff', 'breast.cancer.arff']
# datasets = [ 'arrhythmia.arff', 'balance.scale.arff']

#  'splice.arff',      SO SLOW!
#  'zoo.arff' and 'breast.w.arff' use integer
encoders = [category_encoders.BaseNEncoder(), category_encoders.OneHotEncoder(), category_encoders.BinaryEncoder(), category_encoders.HashingEncoder(), category_encoders.OrdinalEncoder() ]
encoders = [category_encoders.TargetEncoder(), category_encoders.LeaveOneOutEncoder()]
# category_encoders.LeaveOneOutEncoder(), category_encoders.BackwardDifferenceEncoder(), category_encoders.HelmertEncoder(), category_encoders.PolynomialEncoder(), category_encoders.SumEncoder(),

# Initialization
logger = []
scores = None
if os.path.isfile('./output/result.csv'):
    os.remove('./output/result.csv')

# Ok...
warnings.filterwarnings('ignore')

# Loop over datasets, then over encoders, and finally, over the models.
for dataset_name in datasets:
    # X, y = fetch_data(dataset_name, return_X_y=True, local_cache_dir='./datasets/')
    X, y, unique_y, fold_count = arff_loader.load(dataset_name)

    # Debugging
    # print("THIS IS DEBUG CODE!")
    # enc = category_encoders.BackwardDifferenceEncoder()
    # X_transformed = enc.fit_transform(X,y)
    # # scaler = StandardScaler()
    # # X_transformed = scaler.fit_transform(X)
    # # imp = Imputer()
    # # X_transformed = imp.fit_transform(X)
    # cross_validate(estimator=models[1], X=X_transformed, y=y, scoring=['accuracy'], cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=2001), return_train_score=True)
    # print("ok1")

    for encoder in encoders:
        for model in models:
            print('Running:', dataset_name, y.name, fold_count, encoder.__class__.__name__, model.__class__.__name__)
            scores = evaluate(X, y, fold_count, encoder, model, unique_y)

            # Calculate averages over folds
            aggregates = list(np.mean(scores[key]) for key in scores)

            # Log the averages together with metadata
            logger.append([dataset_name, y.name, encoder.__class__.__name__, model.__class__.__name__] + aggregates)

            # Log into csv to have something in case of a crash.
            # If file does not exist, add header.
            result = pd.DataFrame([dataset_name, y.name, encoder.__class__.__name__, model.__class__.__name__] + aggregates).T
            if not os.path.isfile('./output/result.csv'):
                result.to_csv('./output/result.csv', header=['dataset', 'target', 'encoder', 'model'] + list(scores), index=False)
            else:  # else it exists so append without writing the header
                result.to_csv('./output/result.csv', mode='a', header=False, index=False)

# Add a header to the log
result = pd.DataFrame(logger, columns=(['dataset', 'target', 'encoder', 'model'] + list(scores)))


print('Finished')

# Plot the results
# sb.boxplot(data=[logit_test_scores, gnb_test_scores], notch=True)
# plt.xticks([0, 1], ['LogisticRegression', 'GaussianNB'])
# plt.ylabel('Test Accuracy')

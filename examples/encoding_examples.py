"""
Tested to work with scikit-learn 0.20.2
"""

import gc
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

import category_encoders
from examples.source_data.loaders import get_cars_data

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

plt.style.use('ggplot')

__author__ = 'willmcginnis'


def score_models(clf, X, y, encoder, runs=1):
    """
    Takes in a classifier that supports multiclass classification, and X and a y, and returns a cross validation score.

    """

    scores = []

    X_test = None
    for _ in range(runs):
        X_test = encoder().fit_transform(X, y)

        # Some models, like logistic regression, like normalized features otherwise they underperform and/or take a long time to converge.
        # To be rigorous, we should have trained the normalization on each fold individually via pipelines.
        # See grid_search_example to learn how to do it.
        X_test = StandardScaler().fit_transform(X_test)

        scores.append(cross_validate(clf, X_test, y, n_jobs=1, cv=5)['test_score'])
        gc.collect()

    scores = [y for z in [x for x in scores] for y in z]

    return float(np.mean(scores)), float(np.std(scores)), scores, X_test.shape[1]


def main(loader, name):
    """
    Here we iterate through the datasets and score them with a classifier using different encodings.

    """

    scores = []
    raw_scores_ds = {}

    # first get the dataset
    X, y, mapping = loader()

    clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200, random_state=0)

    # try each encoding method available, which works on multiclass problems
    encoders = (set(category_encoders.__all__) - {'WOEEncoder'})  # WoE is currently only for binary targets

    for encoder_name in encoders:
        encoder = getattr(category_encoders, encoder_name)
        start_time = time.time()
        score, stds, raw_scores, dim = score_models(clf, X, y, encoder)
        scores.append([encoder_name, name, dim, score, stds, time.time() - start_time])
        raw_scores_ds[encoder_name] = raw_scores
        gc.collect()

    results = pd.DataFrame(scores, columns=['Encoding', 'Dataset', 'Dimensionality', 'Avg. Score', 'Score StDev', 'Elapsed Time'])

    raw = pd.DataFrame.from_dict(raw_scores_ds)
    ax = raw.plot(kind='box', return_type='axes')
    plt.title('Scores for Encodings on %s Dataset' % (name,))
    plt.ylabel('Score (higher is better)')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return results, raw


if __name__ == '__main__':
    # out, raw = main(get_mushroom_data, 'Mushroom')
    # print(out.sort_values(by=['Dataset', 'Avg. Score']))

    out, raw = main(get_cars_data, 'Cars')
    print(out.sort_values(by=['Dataset', 'Avg. Score']))
    #
    # out, raw = main(get_splice_data, 'Splice')
    # print(out.sort_values(by=['Dataset', 'Avg. Score']))

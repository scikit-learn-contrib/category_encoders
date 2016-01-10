import time
import pandas as pd
import numpy as np
from sklearn import cross_validation, naive_bayes, metrics

import category_encoders
from source_data.loaders import get_cars_data, get_mushroom_data, get_splice_data

__author__ = 'willmcginnis'


def score_models(clf, X, y, runs=10):
    """
    Takes in a classifier that supports multiclass classification, and X and a y, and returns a cross validation score.

    :param clf:
    :param X:
    :param y:
    :return:
    """

    scores = []

    scorer = metrics.make_scorer(metrics.hinge_loss)
    for _ in range(runs):
        scores.append(cross_validation.cross_val_score(clf, X, y, scoring=scorer, n_jobs=-1, cv=5))

    return float(np.mean(scores))


def main(loader, name):
    """
    Here we iterate through the datasets and score them with a classifier using different encodings.
    :return:
    """

    scores = []

    # first get the dataset
    X, y, mapping = loader()
    X = category_encoders.ordinal_encoding(X)

    # create a simple classifier for evaluating encodings (we use bernoulli naive bayed because the data will all be
    # boolean (1/0) by the time the classifier sees it.
    clf = naive_bayes.BernoulliNB()

    # try each encoding method available
    for encoder_name in category_encoders.__all__:
        encoder = category_encoders.__dict__[encoder_name]
        start_time = time.time()
        X_coded = encoder(X)
        score = score_models(clf, X_coded, y)
        scores.append([encoder_name, name, X_coded.shape[1], score, time.time() - start_time])

    results = pd.DataFrame(scores, columns=['Encoding', 'Dataset', 'Dimensionality', 'Avg. Score', 'Elapsed Time'])
    return results

if __name__ == '__main__':
    out = main(get_mushroom_data, 'Mushroom')
    print(out.sort_values(by=['Dataset', 'Avg. Score']))
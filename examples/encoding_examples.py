import time
import gc

import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model

import matplotlib.pyplot as plt
import category_encoders
from examples.source_data.loaders import get_mushroom_data, get_cars_data, get_splice_data

plt.style.use('ggplot')

__author__ = 'willmcginnis'


def score_models(clf, X, y, encoder, runs=1):
    """
    Takes in a classifier that supports multiclass classification, and X and a y, and returns a cross validation score.

    :param clf:
    :param X:
    :param y:
    :return:
    """

    scores = []

    X_test = None
    for _ in range(runs):
        X_test = encoder().fit_transform(X)
        scores.append(cross_validation.cross_val_score(clf, X_test, y, n_jobs=1, cv=5))
        gc.collect()

    scores = [y for z in [x for x in scores] for y in z]

    return float(np.mean(scores)), float(np.std(scores)), scores, X_test.shape[1]


def main(loader, name):
    """
    Here we iterate through the datasets and score them with a classifier using different encodings.
    :return:
    """

    scores = []
    raw_scores_ds = {}

    # first get the dataset
    X, y, mapping = loader()

    clf = linear_model.LogisticRegression()

    # try each encoding method available
    encoders = category_encoders.__all__

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
    plt.title('Scores for Encodings on %s Dataset' % (name, ))
    plt.ylabel('Score (higher better)')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return results, raw

if __name__ == '__main__':
    out, raw = main(get_mushroom_data, 'Mushroom')
    print(out.sort_values(by=['Dataset', 'Avg. Score']))

    out, raw = main(get_cars_data, 'Cars')
    print(out.sort_values(by=['Dataset', 'Avg. Score']))

    out, raw = main(get_splice_data, 'Splice')
    print(out.sort_values(by=['Dataset', 'Avg. Score']))

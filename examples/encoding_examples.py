"""Tested to work with scikit-learn >= 1.6.0."""

import gc
import time
import warnings

import category_encoders
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from examples.source_data.loaders import get_cars_data

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

plt.style.use('ggplot')

__author__ = 'willmcginnis'


def make_evaluation_pipeline(clf, encoder):
    """Build a pipeline that fits preprocessing separately in every CV fold."""
    return Pipeline(
        [
            ('encoder', encoder()),
            ('scaler', StandardScaler()),
            ('classifier', clf),
        ]
    )


def score_models(clf, X, y, encoder, runs=1):
    """Return cross-validation scores for an encoder and classifier pipeline."""
    scores = []
    feature_counts = []

    for _ in range(runs):
        pipeline = make_evaluation_pipeline(clf, encoder)
        cv_results = cross_validate(
            pipeline, X, y, n_jobs=1, cv=5, return_estimator=True
        )
        scores.append(cv_results['test_score'])
        feature_counts.extend(
            len(estimator.named_steps['encoder'].get_feature_names_out())
            for estimator in cv_results['estimator']
        )
        gc.collect()

    scores = [score for run_scores in scores for score in run_scores]

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        scores,
        int(np.mean(feature_counts)),
    )


def main(loader, name):
    """Score the dataset with each encoder that supports multiclass targets."""
    scores = []
    raw_scores_ds = {}

    # first get the dataset
    X, y, mapping = loader()

    clf = linear_model.LogisticRegression(
        solver='lbfgs', max_iter=200, random_state=0
    )

    # try each encoding method available, which works on multiclass problems
    encoders = set(category_encoders.__all__) - {
        'WOEEncoder'
    }  # WoE is currently only for binary targets

    for encoder_name in encoders:
        encoder = getattr(category_encoders, encoder_name)
        start_time = time.time()
        score, stds, raw_scores, dim = score_models(clf, X, y, encoder)
        scores.append([encoder_name, name, dim, score, stds, time.time() - start_time])
        raw_scores_ds[encoder_name] = raw_scores
        gc.collect()

    results = pd.DataFrame(
        scores,
        columns=[
            'Encoding',
            'Dataset',
            'Dimensionality',
            'Avg. Score',
            'Score StDev',
            'Elapsed Time',
        ],
    )

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

"""Tests for the leakage-safe encoding example."""

import category_encoders as ce
import numpy as np
import pandas as pd
from examples.encoding_examples import make_evaluation_pipeline, score_models
from sklearn.linear_model import LogisticRegression


def test_evaluation_pipeline_places_preprocessing_inside_cross_validation():
    """Target encoding and scaling are steps in the estimator passed to CV."""
    pipeline = make_evaluation_pipeline(LogisticRegression(max_iter=200), ce.TargetEncoder)

    assert list(pipeline.named_steps) == ['encoder', 'scaler', 'classifier']
    assert isinstance(pipeline.named_steps['encoder'], ce.TargetEncoder)


def test_score_models_runs_target_encoder_inside_pipeline():
    """A TargetEncoder pipeline returns one finite score from each fold."""
    X = pd.DataFrame(
        {
            'region': ['north', 'south', 'east', 'west'] * 10,
            'value': np.tile([1.0, 2.0, 3.0, 4.0], 10),
        }
    )
    y = np.tile([0, 1, 0, 1], 10)
    classifier = LogisticRegression(max_iter=200, random_state=0)

    mean_score, score_std, scores, feature_count = score_models(
        classifier, X, y, ce.TargetEncoder
    )

    assert len(scores) == 5
    assert np.isfinite(scores).all()
    assert np.isfinite(mean_score)
    assert np.isfinite(score_std)
    assert feature_count == X.shape[1]

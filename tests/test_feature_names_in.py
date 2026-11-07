"""Tests for preserving fitted input column labels."""

import pandas as pd
import pytest
from category_encoders import OrdinalEncoder, SummaryEncoder


@pytest.mark.parametrize('encoder_cls', [OrdinalEncoder, SummaryEncoder])
@pytest.mark.parametrize(
    'columns',
    [
        [0, 'numeric'],
        [(0, 'category'), (1, 'numeric')],
        ['category', 'numeric'],
        [0, 1],
    ],
    ids=['mixed', 'tuples', 'strings', 'integers'],
)
def test_feature_names_in_preserve_column_labels(encoder_cls, columns):
    """Input feature names can select the original fitted DataFrame columns."""
    X = pd.DataFrame(
        [['a', 1], ['a', 2], ['b', 3], ['b', 4]],
        columns=pd.Index(columns, tupleize_cols=False),
    )
    encoder = encoder_cls(cols=[columns[0]]).fit(X, [1.0, 2.0, 3.0, 4.0])

    feature_names = encoder.get_feature_names_in()

    assert feature_names.ndim == 1
    assert feature_names.tolist() == columns
    pd.testing.assert_frame_equal(X.loc[:, feature_names], X)

"""Regression tests for collisions between generated category column names."""

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    'encoder_class', [encoders.OneHotEncoder, encoders.RankHotEncoder, encoders.MultiHotEncoder]
)
@pytest.mark.parametrize('use_cat_names', [False, True])
@pytest.mark.parametrize('drop_invariant', [False, True])
def test_generated_names_with_literal_suffixes(encoder_class, use_cat_names, drop_invariant):
    """Overlapping prefixes and literal suffixes preserve every encoded feature."""
    X = pd.DataFrame(
        {
            'c': ['x_a', 'x_a#', 'x_z'],
            'c_x': ['a', 'a#', 'z'],
            'keep': [10, 20, 30],
        },
        index=[3, 1, 3],
    )
    names = (
        ['c_x_a', 'c_x_a#', 'c_x_z', 'c_x_a##', 'c_x_a###', 'c_x_z#']
        if use_cat_names
        else ['c_1', 'c_2', 'c_3', 'c_x_1', 'c_x_2', 'c_x_3']
    )
    values = (
        np.tril(np.ones((3, 3), dtype=int))
        if encoder_class is encoders.RankHotEncoder
        else np.eye(3, dtype=int)
    )
    expected = pd.DataFrame(np.tile(values, (1, 2)), columns=names, index=X.index)
    expected['keep'] = X['keep']
    if drop_invariant and encoder_class is encoders.RankHotEncoder:
        expected = expected.drop(columns=[names[0], names[3]])

    encoder = encoder_class(use_cat_names=use_cat_names, drop_invariant=drop_invariant)
    out = encoder.fit_transform(X)

    pd.testing.assert_frame_equal(out, expected, check_dtype=False)
    assert out.columns.is_unique
    assert_array_equal(encoder.get_feature_names_out(), expected.columns)
    pd.testing.assert_frame_equal(
        encoder.transform(X.iloc[::-1]), expected.iloc[::-1], check_dtype=False
    )
    if not drop_invariant:
        if encoder_class is not encoders.MultiHotEncoder:
            pd.testing.assert_frame_equal(encoder.inverse_transform(out), X)
        pd.testing.assert_frame_equal(encoder.fit_transform(X), expected, check_dtype=False)


@pytest.mark.parametrize(
    'encoder_class, options, marker, unseen',
    [
        (encoders.OneHotEncoder, {'handle_unknown': 'indicator'}, '-1', 'new'),
        (encoders.MultiHotEncoder, {'handle_unknown': 'indicator'}, '-1', 'new'),
        (encoders.OneHotEncoder, {'handle_missing': 'indicator'}, 'nan', np.nan),
        (encoders.MultiHotEncoder, {'handle_missing': 'indicator'}, '-2', np.nan),
    ],
)
def test_indicator_names_with_literal_suffixes(encoder_class, options, marker, unseen):
    """Indicators remain distinct from literal items ending in one or more '#'."""
    items = [marker, marker + '#', marker + '##']
    encoder = encoder_class(use_cat_names=True, **options).fit(pd.DataFrame({'c': items}))
    X = pd.DataFrame({'c': items + [unseen]}, index=[4, 2, 4, 1])
    expected = pd.DataFrame(
        np.eye(4, dtype=int),
        columns=[f'c_{marker}' + '#' * i for i in range(4)],
        index=X.index,
    )

    pd.testing.assert_frame_equal(encoder.transform(X), expected, check_dtype=False)
    assert_array_equal(encoder.get_feature_names_out(), expected.columns)


@pytest.mark.parametrize(
    'encoder_class, values',
    [
        (encoders.OneHotEncoder, [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        (encoders.MultiHotEncoder, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_missing_value_names_with_literal_suffixes(encoder_class, values):
    """A learned missing value cannot collide with literal 'nan' or 'nan#'."""
    X = pd.DataFrame({'c': ['nan', np.nan, 'nan#']})
    encoder = encoder_class(use_cat_names=True)
    expected = pd.DataFrame(values, columns=['c_nan', 'c_nan#', 'c_nan##'])

    pd.testing.assert_frame_equal(encoder.fit_transform(X), expected, check_dtype=False)
    assert_array_equal(encoder.get_feature_names_out(), expected.columns)

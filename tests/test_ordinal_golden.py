"""Golden parity tests for OrdinalEncoder.ordinal_encoding.

The GOLDEN outputs below were captured from the pre-refactor implementation at
upstream master b2b8691 (the #446 restructure of ``ordinal_encoding`` into
named steps). They pin the contract of the refactor: fitted mappings and
encodings must stay identical. If a future change deliberately alters encoding
behavior, regenerate the goldens and review the diff explicitly — do not edit
them casually.

Scenario coverage: fit-side mapping construction (NaN last, ordered and
unordered category dtypes, index_start, -2 missing sentinel), transform-side
mapping application (unknown -> -1, late missing -> -2, return_nan, supplied
dict/series mappings), and the full universal ``create_dataset`` fixture.
"""

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest
from numpy import inf, nan  # bare names used in the GOLDEN literal below

import tests.helpers as th

GOLDEN = {
    'default': {
        'mapping': [
            {
                'col': 'str_col',
                'data_type': 'object',
                'mapping': {'a': 1, 'b': 2, 'c': 3, 'nan': -2},
            },
            {'col': 'nan_col', 'data_type': 'object', 'mapping': {'nan': 3, 'x': 1, 'y': 2}},
            {
                'col': 'cat_col',
                'data_type': 'category',
                'mapping': {'A': 1, 'B': 2, 'C': 3, 'nan': -2},
            },
            {'col': 'ord_col', 'data_type': 'category', 'mapping': {'A': 1, 'B': 2, 'nan': -2}},
        ],
        'transform': [
            [1, 1, 1, 2, 3],
            [2, 2, 2, 1, 1],
            [1, 3, 1, 2, 2],
            [3, 1, 3, 1, 1],
            [2, 2, 2, 2, 3],
        ],
    },
    'index_start_zero': {
        'mapping': [
            {
                'col': 'str_col',
                'data_type': 'object',
                'mapping': {'a': 0, 'b': 1, 'c': 2, 'nan': -2},
            },
            {'col': 'nan_col', 'data_type': 'object', 'mapping': {'nan': 2, 'x': 0, 'y': 1}},
            {
                'col': 'cat_col',
                'data_type': 'category',
                'mapping': {'A': 0, 'B': 1, 'C': 2, 'nan': -2},
            },
            {'col': 'ord_col', 'data_type': 'category', 'mapping': {'A': 0, 'B': 1, 'nan': -2}},
        ],
        'transform': [
            [0, 0, 0, 1, 3],
            [1, 1, 1, 0, 1],
            [0, 2, 0, 1, 2],
            [2, 0, 2, 0, 1],
            [1, 1, 1, 1, 3],
        ],
    },
    'missing_late': {
        'mapping': [
            {
                'col': 'str_col',
                'data_type': 'object',
                'mapping': {'a': 1, 'b': 2, 'c': 3, 'nan': -2},
            },
            {'col': 'nan_col', 'data_type': 'object', 'mapping': {'nan': -2, 'x': 1, 'y': 2}},
            {
                'col': 'cat_col',
                'data_type': 'category',
                'mapping': {'A': 1, 'B': 2, 'C': 3, 'nan': -2},
            },
            {'col': 'ord_col', 'data_type': 'category', 'mapping': {'A': 1, 'B': 2, 'nan': -2}},
        ],
        'transform': [
            [1, 1, 1, 2, 3],
            [2, -2, 2, 1, 1],
            [1, 1, 1, 2, 2],
            [3, 1, 3, 1, 1],
            [2, -2, 2, 2, 3],
        ],
    },
    'missing_return_nan': {
        'mapping': [
            {
                'col': 'str_col',
                'data_type': 'object',
                'mapping': {'a': 1, 'b': 2, 'c': 3, 'nan': -2},
            },
            {'col': 'nan_col', 'data_type': 'object', 'mapping': {'nan': -2, 'x': 1, 'y': 2}},
            {
                'col': 'cat_col',
                'data_type': 'category',
                'mapping': {'A': 1, 'B': 2, 'C': 3, 'nan': -2},
            },
            {'col': 'ord_col', 'data_type': 'category', 'mapping': {'A': 1, 'B': 2, 'nan': -2}},
        ],
        'transform': [
            [1.0, 1.0, 1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, nan, 1.0, 2.0, 2.0],
            [3.0, 1.0, 3.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 3.0],
        ],
    },
    'supplied_dict': {'transform': [[1.0], [2.0], [-1.0], [1.0], [-1.0]]},
    'supplied_series': {'transform': [[1], [2], [7], [1], [3]]},
    'universal_fixture': {
        'mapping': [
            {
                'col': 'unique_str',
                'data_type': 'object',
                'mapping': {
                    '0': 1,
                    '1': 2,
                    '10': 11,
                    '11': 12,
                    '2': 3,
                    '3': 4,
                    '4': 5,
                    '5': 6,
                    '6': 7,
                    '7': 8,
                    '8': 9,
                    '9': 10,
                    'nan': -2,
                },
            },
            {'col': 'invariant', 'data_type': 'object', 'mapping': {'A': 1, 'nan': -2}},
            {
                'col': 'underscore',
                'data_type': 'object',
                'mapping': {'A': 1, 'B_b': 3, 'C_c_c': 2, 'nan': -2},
            },
            {'col': 'none', 'data_type': 'object', 'mapping': {'A': 1, 'B': 2, 'C': 3, 'nan': 4}},
            {'col': 'extra', 'data_type': 'object', 'mapping': {'A': 3, 'B': 2, 'C': 1, 'nan': -2}},
            {
                'col': 'categorical',
                'data_type': 'category',
                'mapping': {'A': 3, 'B': 2, 'C': 1, 'nan': -2},
            },
            {
                'col': 'na_categorical',
                'data_type': 'category',
                'mapping': {'A': 1, 'B': 2, 'C': 3, 'nan': 4},
            },
            {
                'col': 'categorical_int',
                'data_type': 'category',
                'mapping': {1.0: 2, 2.0: 3, 3.0: 1, 'nan': -2},
            },
        ],
        'transform': [
            [0.5864317642276706, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.9355421568865638, -inf, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            [0.6417434471292205, inf, 2.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 3.0, 3.0],
            [0.37577403647270746, -inf, 3.0, 4.0, 1.0, 3.0, 4.0, 2.0, 1.0, 1.0, 3.0],
            [0.4423799384321432, 3.141592653589793, 4.0, 5.0, 1.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0],
            [0.1369615639667422, 0.0, 5.0, 6.0, 1.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0],
            [0.41791811086718744, -inf, 6.0, 7.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0],
            [0.09629252921986808, nan, 7.0, 8.0, 1.0, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0],
            [0.1732005459327527, inf, 8.0, 9.0, 1.0, 2.0, 4.0, 3.0, 2.0, 2.0, 3.0],
            [0.22470632478416086, 0.0, 9.0, 10.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            [0.6759283730312308, inf, 10.0, 11.0, 1.0, 1.0, 2.0, 3.0, 1.0, 3.0, 1.0],
            [0.07857232873044162, inf, 11.0, 12.0, 1.0, 2.0, 3.0, 2.0, 2.0, 1.0, 2.0],
        ],
    },
    'unknown_error': {'message': 'Unexpected categories found in column str_col', 'raised': True},
    'unknown_value': {
        'transform': [
            [1.0, 1.0, 1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 1.0, 1.0],
            [-1.0, 3.0, 1.0, 2.0, 2.0],
            [3.0, 1.0, 3.0, 1.0, 1.0],
            [-1.0, 2.0, 2.0, 2.0, 3.0],
        ]
    },
}


def _small_frame():
    """Fixture covering string, missing, unordered/ordered categorical, and int columns."""
    return pd.DataFrame(
        {
            'str_col': ['a', 'b', 'a', 'c', 'b'],
            'nan_col': ['x', 'y', np.nan, 'x', 'y'],
            'cat_col': pd.Categorical(['A', 'B', 'A', 'C', 'B'], categories=['C', 'A', 'B']),
            'ord_col': pd.Categorical(
                ['B', 'A', 'B', 'A', 'B'], categories=['A', 'B'], ordered=True
            ),
            'int_col': [3, 1, 2, 1, 3],
        }
    )


def _categorical_nan_frame():
    """Fixture for supplied mappings on a categorical column with missing values."""
    return pd.DataFrame(
        {'cat_nan': pd.Categorical(['A', 'B', np.nan, 'A', 'C'], categories=['A', 'B', 'C'])}
    )


def _unseen_frame(X):
    """Copy of X with unseen categories in str_col."""
    X_unseen = X.copy()
    X_unseen.loc[:, 'str_col'] = ['a', 'b', 'd', 'c', 'e']
    return X_unseen


def _canonical_mapping(mapping):
    """Category -> code dict with NaN keys canonicalized to the string 'nan'."""
    return {'nan' if pd.isna(key) else key: int(value) for key, value in mapping.to_dict().items()}


def _assert_golden_mapping(actual_switches, expected_switches):
    assert len(actual_switches) == len(expected_switches)
    for switch, expected in zip(actual_switches, expected_switches, strict=True):
        assert switch['col'] == expected['col']
        assert switch['data_type'] == expected['data_type']
        assert _canonical_mapping(switch['mapping']) == expected['mapping']


def _assert_golden_values(actual_frame, expected_rows):
    rows = actual_frame.to_numpy().tolist()
    assert len(rows) == len(expected_rows)
    for r, (actual_row, expected_row) in enumerate(zip(rows, expected_rows, strict=True)):
        assert len(actual_row) == len(expected_row)
        for c, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row, strict=True)
        ):
            if pd.isna(expected_value):
                assert pd.isna(actual_value), (
                    f'row {r}, col {c}: expected NaN, got {actual_value!r}'
                )
            else:
                assert actual_value == expected_value, (
                    f'row {r}, col {c}: expected {expected_value!r}, got {actual_value!r}'
                )


@pytest.mark.parametrize(
    ('encoder_kwargs', 'scenario'),
    [
        ({}, 'default'),
        ({'index_start': 0}, 'index_start_zero'),
        ({'handle_missing': 'return_nan'}, 'missing_return_nan'),
    ],
)
def test_golden_fit_mapping_and_encoding(encoder_kwargs, scenario):
    """Fitted mappings and encodings match the pre-refactor golden outputs."""
    golden = GOLDEN[scenario]
    X = _small_frame()
    encoder = encoders.OrdinalEncoder(**encoder_kwargs).fit(X)

    _assert_golden_mapping(encoder.mapping, golden['mapping'])
    _assert_golden_values(encoder.transform(X), golden['transform'])


def test_golden_unknown_and_missing_at_transform():
    """Unseen categories encode as -1 and late missing values as -2, per the goldens."""
    X = _small_frame()
    encoder = encoders.OrdinalEncoder().fit(X)
    _assert_golden_values(encoder.transform(_unseen_frame(X)), GOLDEN['unknown_value']['transform'])

    # NaN absent at fit, present at transform: encoded as the -2 sentinel.
    X_no_nan = X.copy()
    X_no_nan.loc[:, 'nan_col'] = ['x', 'y', 'x', 'x', 'y']
    encoder = encoders.OrdinalEncoder().fit(X_no_nan)
    X_nan_at_transform = X_no_nan.copy()
    X_nan_at_transform.loc[:, 'nan_col'] = ['x', np.nan, 'x', 'x', np.nan]
    _assert_golden_mapping(encoder.mapping, GOLDEN['missing_late']['mapping'])
    _assert_golden_values(
        encoder.transform(X_nan_at_transform), GOLDEN['missing_late']['transform']
    )


def test_golden_unknown_error():
    """handle_unknown='error' raises on unseen categories, per the golden message."""
    X = _small_frame()
    encoder = encoders.OrdinalEncoder(handle_unknown='error').fit(X)

    with pytest.raises(ValueError, match=GOLDEN['unknown_error']['message']):
        encoder.transform(_unseen_frame(X))


@pytest.mark.parametrize(
    ('supplied_mapping', 'scenario'),
    [
        ([{'col': 'cat_nan', 'mapping': {None: 0, 'A': 1, 'B': 2}}], 'supplied_dict'),
        (
            [{'col': 'cat_nan', 'mapping': pd.Series({np.nan: 7, 'A': 1, 'B': 2, 'C': 3})}],
            'supplied_series',
        ),
    ],
)
def test_golden_supplied_mapping_encoding(supplied_mapping, scenario):
    """Supplied dict/series mapping encodings match the pre-refactor golden outputs."""
    encoder = encoders.OrdinalEncoder(mapping=supplied_mapping)
    _assert_golden_values(
        encoder.fit_transform(_categorical_nan_frame()), GOLDEN[scenario]['transform']
    )


def test_golden_universal_fixture():
    """Encodings of the universal fixture (create_dataset) match the pre-refactor goldens."""
    X = th.create_dataset(n_rows=12)
    encoder = encoders.OrdinalEncoder().fit(X)

    golden = GOLDEN['universal_fixture']
    _assert_golden_mapping(encoder.mapping, golden['mapping'])
    _assert_golden_values(encoder.transform(X), golden['transform'])

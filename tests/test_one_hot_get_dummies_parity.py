"""Golden parity tests for OneHotEncoder.get_dummies (GH #362).

The allocation pattern of ``get_dummies`` was rewritten from a per-column
concat accumulator plus final reindex to a single concat over pre-ordered
dummy blocks. ``reference_get_dummies`` below is the pre-fix implementation,
kept verbatim as the parity reference: every fixture asserts the current
implementation produces the exact same frame (values, column order, dtypes,
index) as the reference, and raises the same exception type where the old
code raised.
"""

import numpy as np
import pandas as pd
import pytest
from category_encoders import OneHotEncoder


def reference_get_dummies(X_in: pd.DataFrame, mapping: list) -> pd.DataFrame:
    """Pre-#362 accumulator implementation, kept as the parity reference."""
    X = X_in.copy(deep=True)

    cols = X.columns.tolist()

    for switch in mapping:
        col = switch.get('col')
        mod = switch.get('mapping')

        base_df = mod.reindex(X[col].fillna(-2))
        base_df = base_df.set_index(X.index)
        X = pd.concat([base_df, X], axis=1)

        old_column_index = cols.index(col)
        cols[old_column_index : old_column_index + 1] = mod.columns

    return X.reindex(columns=cols)


def assert_frames_identical(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Assert bit-identical values, column order, dtypes, and index."""
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.dtypes.equals(expected.dtypes)
    assert actual.index.equals(expected.index)


@pytest.fixture(name='mixed_train')
def mixed_train_fixture() -> pd.DataFrame:
    """Mixed-dtype frame with missing values and a passthrough column."""
    return pd.DataFrame(
        {
            'Str': ['a', 'b', 'b', 'c', 'a', 'b'],
            'Numbers': [1.0, 2.0, 3.0, np.nan, 2.0, 1.0],
            'Ints': [1, 2, 3, 4, 5, 6],
            'Bools': [True, False, True, True, False, True],
            'Pass': ['x', 'y', 'z', 'x', 'y', 'z'],
        }
    )


def fit_on(frame: pd.DataFrame, cols: list, **kwargs) -> OneHotEncoder:
    """Fit a OneHotEncoder on ``frame`` and return it."""
    enc = OneHotEncoder(cols=cols, **kwargs)
    enc.fit(frame)
    return enc


def test_matches_reference_on_mixed_dtypes(mixed_train):
    """Encoded string, float, int, and bool columns with a passthrough column."""
    enc = fit_on(mixed_train, cols=['Str', 'Numbers', 'Ints', 'Bools'])
    codes = enc.ordinal_encoder.transform(mixed_train)
    assert_frames_identical(enc.get_dummies(codes), reference_get_dummies(codes, enc.mapping))


def test_matches_reference_with_unknown_and_missing_values(mixed_train):
    """Codes absent from the mapping and missing values imputed with -2."""
    enc = fit_on(mixed_train, cols=['Str', 'Numbers', 'Ints'])
    test_frame = pd.DataFrame(
        {
            'Str': ['a', 'unknown', np.nan, 'c'],
            'Numbers': [1.0, np.nan, 9.0, 3.0],
            'Ints': [7, 2, np.nan, 4],
            'Bools': [True, False, True, True],
            'Pass': ['z', 'y', 'x', 'x'],
        }
    )
    codes = enc.ordinal_encoder.transform(test_frame)
    assert_frames_identical(enc.get_dummies(codes), reference_get_dummies(codes, enc.mapping))


@pytest.mark.parametrize(
    'handle_unknown,handle_missing',
    [
        ('value', 'value'),
        ('return_nan', 'value'),
        ('value', 'return_nan'),
        ('return_nan', 'return_nan'),
    ],
)
def test_matches_reference_across_handle_configs(mixed_train, handle_unknown, handle_missing):
    """NaN-producing reindex rows for every non-error handle config."""
    enc = fit_on(
        mixed_train,
        cols=['Str', 'Numbers', 'Ints'],
        handle_unknown=handle_unknown,
        handle_missing=handle_missing,
    )
    test_frame = mixed_train.copy()
    test_frame.loc[0, 'Str'] = 'unseen'  # unknown category
    codes = enc.ordinal_encoder.transform(test_frame)
    assert_frames_identical(enc.get_dummies(codes), reference_get_dummies(codes, enc.mapping))


def test_matches_reference_on_high_cardinality_column():
    """Many dummy blocks per column, reporter-like cardinality mix."""
    frame = pd.DataFrame(
        {
            'Wide': [f'cat_{i % 137:03d}' for i in range(275)],
            'Narrow': ['n0', 'n1'] * 137 + ['n0'],
            'Keep': range(275),
        }
    )
    enc = fit_on(frame, cols=['Wide', 'Narrow'])
    codes = enc.ordinal_encoder.transform(frame)
    assert_frames_identical(enc.get_dummies(codes), reference_get_dummies(codes, enc.mapping))


def test_preserves_input_frame_and_index(mixed_train):
    """The input frame is not mutated and a non-default index is kept."""
    enc = fit_on(mixed_train, cols=['Str', 'Numbers'])
    frame = mixed_train.copy()
    frame.index = [10, 3, 7, 1, 42, 0]
    snapshot = frame.copy(deep=True)

    codes = enc.ordinal_encoder.transform(frame)
    out = enc.get_dummies(codes)

    assert_frames_identical(out, reference_get_dummies(codes, enc.mapping))
    pd.testing.assert_frame_equal(frame, snapshot)
    assert out.index.equals(frame.index)


def test_untouched_frame_passes_through_unchanged():
    """An empty mapping returns the frame content unchanged."""
    frame = pd.DataFrame({'Pass': ['x', 'y'], 'Other': [1, 2]})
    enc = fit_on(frame, cols=[])

    out = enc.get_dummies(frame.copy(deep=True))

    assert_frames_identical(out, frame)


def test_missing_mapping_column_raises_key_error(mixed_train):
    """A fitted column missing from the input raises KeyError."""
    enc = fit_on(mixed_train, cols=['Str'])
    dropped = mixed_train.drop(columns=['Str'])

    with pytest.raises(KeyError):
        enc.get_dummies(dropped)


def test_dummy_name_collision_with_existing_column_raises():
    """A dummy name colliding with a passthrough column raises ValueError."""
    # encoding 'A' produces dummies named 'A_1'/'A_2'; the passthrough column
    # 'A_1' collides with the first dummy. The old code hit the same
    # ValueError (same message) in its final reindex, during fit's internal
    # transform.
    frame = pd.DataFrame({'A': ['a', 'b', 'b'], 'A_1': ['p', 'q', 'r']})
    enc = OneHotEncoder(cols=['A'])

    with pytest.raises(ValueError):
        enc.fit(frame)


def test_encoded_column_named_like_another_dummy_encodes_fully():
    """Encoded 'A' next to encoded 'A_1' (which is a dummy name of 'A').

    The old accumulator crashed mid-loop ('Index data must be 1-dimensional')
    because its growing frame held the freshly created 'A_1' dummy next to
    the still-unencoded 'A_1' column. The single-concat build never forms
    that corrupted intermediate, so both columns encode fully with unique
    output labels. Documented in the PR as an intentional divergence.
    """
    frame = pd.DataFrame({'A': ['a', 'b', 'b'], 'A_1': ['a', 'b', 'b']})
    enc = fit_on(frame, cols=['A', 'A_1'])

    out = enc.get_dummies(enc.ordinal_encoder.transform(frame))

    assert not out.columns.has_duplicates
    assert out.shape == (3, 4)


def test_passthrough_named_like_a_dummy_without_collision_is_fine(mixed_train):
    """A passthrough named like a dummy stays legal without its source column."""
    # 'A_1' passthrough only collides when a column 'A' is encoded; with no
    # such column the frame must pass through without any error
    frame = mixed_train.rename(columns={'Pass': 'A_1'})
    enc = fit_on(frame, cols=['Str', 'Numbers'])

    codes = enc.ordinal_encoder.transform(frame)
    assert_frames_identical(enc.get_dummies(codes), reference_get_dummies(codes, enc.mapping))

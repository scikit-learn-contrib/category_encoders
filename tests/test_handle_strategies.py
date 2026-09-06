"""Tests for int and callable handle_unknown / handle_missing strategies.

The contracts tested here:

- a numeric scalar is used directly as the encoded value for unknown
  labels / missing values that were not seen at fit time
- a callable ``fn(value, mapping)`` returns the encoded value; supervised
  encoders evaluate it once per column when the mapping is finalized during
  fit, OrdinalEncoder evaluates it per unseen value at transform time
"""

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest

# Ordinal plus every supervised encoder that finalizes its mapping through
# utils.finalize_encoding_mapping (SummaryEncoder delegates to QuantileEncoder).
SCALAR_ENCODERS = [
    'OrdinalEncoder',
    'TargetEncoder',
    'MEstimateEncoder',
    'JamesSteinEncoder',
    'WOEEncoder',
    'QuantileEncoder',
    'SummaryEncoder',
    'GLMMEncoder',
]


@pytest.fixture(name='train_frame')
def fixture_train_frame():
    """A small three-category frame with a repeated label."""
    return pd.DataFrame({'col': ['a', 'b', 'c', 'a', 'b', 'c']})

@pytest.fixture(name='target')
def fixture_target():
    """A binary target aligned with train_frame."""
    return pd.Series([1, 0, 1, 0, 1, 0], name='y')


def make_unseen_and_missing():
    """Rows: known, unseen label, known, missing value (never seen at fit)."""
    return pd.DataFrame({'col': ['a', 'unseen label', 'a', np.nan]})


@pytest.mark.parametrize('encoder_name', SCALAR_ENCODERS)
def test_int_unknown_and_missing(encoder_name, train_frame, target):
    """Every supporting encoder honors int fills for unknown and missing rows."""
    encoder = getattr(encoders, encoder_name)(handle_unknown=7, handle_missing=13)
    X_test = make_unseen_and_missing()
    result = encoder.fit(train_frame, target).transform(X_test)

    out_cols = [col for col in result.columns if col.startswith('col')]
    known = result[out_cols].iloc[0]
    unknown = result[out_cols].iloc[1]
    missing = result[out_cols].iloc[3]

    assert (unknown == 7).all(), f'{encoder_name}: unknown rows must fill with 7'
    assert (missing == 13).all(), f'{encoder_name}: missing rows must fill with 13'
    assert not (known == 7).all(), f'{encoder_name}: known rows must keep their encoding'


@pytest.mark.parametrize('encoder_name', SCALAR_ENCODERS)
def test_float_scalar(encoder_name, train_frame, target):
    """Float scalars are accepted like ints."""
    encoder = getattr(encoders, encoder_name)(handle_unknown=-0.5, handle_missing=0.25)
    X_test = make_unseen_and_missing()
    result = encoder.fit(train_frame, target).transform(X_test)

    out_cols = [col for col in result.columns if col.startswith('col')]
    assert (result[out_cols].iloc[1] == -0.5).all()
    assert (result[out_cols].iloc[3] == 0.25).all()


def test_int_fill_non_negative_for_boosting(train_frame):
    """Non-negative fills keep the output usable for models like LightGBM."""
    encoder = encoders.OrdinalEncoder(handle_unknown=100, handle_missing=200)
    X_test = pd.DataFrame({'col': ['a', 'unseen label', np.nan]})
    result = encoder.fit(train_frame).transform(X_test)

    assert result['col'].min() >= 0


def test_callable_unknown_ordinal_receives_value_and_mapping(train_frame):
    """OrdinalEncoder evaluates the callable per unseen value with the fitted mapping."""
    calls = []

    def fn(value, mapping):
        calls.append((value, mapping))
        return 555

    encoder = encoders.OrdinalEncoder(handle_unknown=fn)
    X_test = pd.DataFrame({'col': ['a', 'zzz extra']})
    result = encoder.fit(train_frame).transform(X_test)

    assert result['col'].tolist() == [1.0, 555.0]
    assert len(calls) == 1
    assert calls[0][0] == 'zzz extra', 'the callable receives the raw unseen label'
    assert 'a' in calls[0][1].index, 'the callable receives the fitted mapping'


def test_callable_unknown_ordinal_can_read_the_mapping(train_frame):
    """A callable may look the unseen label up in (or beyond) the fitted mapping."""
    encoder = encoders.OrdinalEncoder(
        handle_unknown=lambda value, mapping: mapping.max() + 10
    )
    X_test = pd.DataFrame({'col': ['a', 'zz']})
    result = encoder.fit(train_frame).transform(X_test)

    assert result['col'].tolist() == [1, 13]  # max fitted label is 3 -> 3 + 10


def test_callable_missing_ordinal_receives_nan(train_frame):
    """OrdinalEncoder evaluates the missing callable with value=nan and the fitted mapping."""
    seen_values = []

    def fn(value, mapping):
        seen_values.append(value)
        return -77

    encoder = encoders.OrdinalEncoder(handle_missing=fn)
    X_test = pd.DataFrame({'col': ['a', np.nan]})
    result = encoder.fit(train_frame).transform(X_test)

    assert result['col'].tolist() == [1.0, -77.0]
    assert len(seen_values) > 0
    assert all(np.isnan(value) for value in seen_values)


def test_callable_unknown_target(train_frame, target):
    """TargetEncoder evaluates the unknown callable once per column at fit time."""
    call_count = []

    def fn(value, mapping):
        call_count.append(value)
        return mapping.max() + 100

    encoder = encoders.TargetEncoder(handle_unknown=fn)
    X_test = pd.DataFrame({'col': ['a', 'zz']})
    result = encoder.fit(train_frame, target).transform(X_test)

    assert result['col'].tolist() == [0.5, 103.0]  # max ordinal code is 3 -> 3 + 100
    assert len(call_count) == 1, 'the callable is evaluated once per column during fit'


def test_callable_missing_target(train_frame, target):
    """TargetEncoder evaluates the missing callable once per column at fit time."""
    encoder = encoders.TargetEncoder(
        handle_unknown=42, handle_missing=lambda value, mapping: mapping.max() + 200
    )
    X_test = pd.DataFrame({'col': ['a', np.nan]})
    result = encoder.fit(train_frame, target).transform(X_test)

    assert result['col'].tolist() == [0.5, 203.0]  # max ordinal code is 3 -> 3 + 200


def test_callable_recomputed_on_refit(train_frame, target):
    """A second fit re-evaluates callables against the new mapping."""
    encoder = encoders.TargetEncoder(handle_unknown=lambda value, mapping: mapping.max() + 100)
    encoder.fit(train_frame, target)
    assert encoder.transform(pd.DataFrame({'col': ['zz']}))['col'].iloc[0] == 103.0

    bigger = pd.DataFrame({'col': ['a', 'b', 'c', 'd', 'e', 'f']})
    encoder.fit(bigger, target)
    assert encoder.transform(pd.DataFrame({'col': ['zz']}))['col'].iloc[0] == 106.0


@pytest.mark.parametrize(
    'encoder_name,kwargs',
    [
        ('OrdinalEncoder', {'handle_unknown': 1}),
        ('OrdinalEncoder', {'handle_missing': 1}),
        ('TargetEncoder', {'handle_unknown': 0.5}),
        ('TargetEncoder', {'handle_missing': 0.5}),
    ],
)
def test_scalar_collision_is_rejected(encoder_name, kwargs, train_frame, target):
    """A scalar that collides with a generated label would blur unknown/missing rows."""
    encoder = getattr(encoders, encoder_name)(**kwargs)
    with pytest.raises(ValueError, match='collides'):
        encoder.fit(train_frame, target)


@pytest.mark.parametrize(
    'encoder_cls,expected',
    [
        (encoders.OrdinalEncoder, [-1.0, -2.0]),
        (encoders.TargetEncoder, [0.5, 0.5]),
    ],
)
def test_string_strategy_defaults_unchanged(encoder_cls, expected, train_frame, target):
    """The historical -1/-2 (and prior-mean) defaults are untouched."""
    encoder = encoder_cls()
    X_test = pd.DataFrame({'col': ['unseen label', np.nan]})
    result = encoder.fit(train_frame, target).transform(X_test)

    assert result['col'].tolist() == expected


def test_scalar_and_string_strategies_combine(train_frame, target):
    """A scalar handle_missing coexists with the default string handle_unknown."""
    encoder = encoders.OrdinalEncoder(handle_missing=99)
    X_test = make_unseen_and_missing()
    result = encoder.fit(train_frame).transform(X_test)

    assert result['col'].tolist() == [1.0, -1.0, 1.0, 99.0]


def test_callable_result_must_be_scalar(train_frame, target):
    """A callable returning a non-scalar is rejected instead of corrupting the mapping."""
    encoder = encoders.TargetEncoder(handle_unknown=lambda value, mapping: [1, 2])
    with pytest.raises(ValueError, match='numeric scalar'):
        encoder.fit(train_frame, target)

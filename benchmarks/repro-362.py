"""Memory repro for GH issue #362: OneHotEncoder transform peak memory.

Measures tracemalloc peaks of OneHotEncoder fit/transform and of the
``get_dummies`` phase in isolation, then compares the implementation in this
checkout against two allocation-pattern variants (all outputs are asserted
identical):

* current:       whatever this checkout ships (the single-concat block build
                 introduced by the #362 fix; upstream master used a
                 per-column ``pd.concat`` accumulator plus a final reindex)
* single-concat: build per-column dummy blocks in final order, one
                 ``pd.concat``, no post-concat reindex
* bool-growth:   v1.3.0-style in-place column growth (default config only)

Usage: python3 benchmarks/repro-362.py [--rows 100000] [--cols 10] [--cats 20]
[--scenario-b]

Scenario A default (100k x 10 cols x 20 cats) is reporter-like; scenario B
(20k x 2 cols x 500 cats) is high-cardinality. Run on the base commit and on
the fix branch; the get_dummies-phase and whole-transform peaks are the
before/after numbers for the PR.

Measured on Python 3.13 / pandas 2.2.3 / numpy 2.3.5, upstream master
(b2b8691): get_dummies phase 610.4 MB (A) / 459.1 MB (B), whole transform
625.7 / 459.7 MB.
"""

import argparse
import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from category_encoders import OneHotEncoder


def build_frame(rows: int, n_cols: int, n_cats: int, seed: int = 0) -> pd.DataFrame:
    """Build a random categorical frame for the measured scenarios."""
    rng = np.random.default_rng(seed)
    categories = np.array([f'cat_{i:04d}' for i in range(n_cats)])
    data = {f'c{idx}': rng.choice(categories, size=rows) for idx in range(n_cols)}
    return pd.DataFrame(data)


def peak_mb(fn, *args, **kwargs) -> float:
    """Run ``fn`` inside a fresh tracemalloc window and return the peak in MB."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if result is not None:
        del result
    return peak / 1024 / 1024


def dummies_single_concat(X_in: pd.DataFrame, mapping) -> pd.DataFrame:
    """Single-concat variant: per-column dummy blocks in final order, one concat.

    Blocks follow the input column order (encoded columns expand in place), so
    the concat result is already in final order and no post-concat reindex copy
    is needed. The duplicate-label guard preserves master's ValueError on
    dummy-name collisions (master hits the same error in its final reindex).
    """
    blocks_by_col = {switch.get('col'): switch.get('mapping') for switch in mapping}

    blocks = []
    for col in X_in.columns:
        if col in blocks_by_col:
            mod = blocks_by_col[col]
            base_df = mod.reindex(X_in[col].fillna(-2))
            blocks.append(base_df.set_index(X_in.index))
        else:
            blocks.append(X_in[[col]])

    result = pd.concat(blocks, axis=1)
    if result.columns.has_duplicates:
        raise ValueError('cannot reindex on an axis with duplicate labels')
    return result


def dummies_bool_growth(X_in: pd.DataFrame, mapping) -> pd.DataFrame:
    """Bool-growth variant: v1.3.0-style in-place growth (mutates X_in).

    Exact only for the default handle_unknown='value' / handle_missing='value'
    config (all codes present in the mapping index, int64 output); measured for
    comparison. Grows the frame by per-column assignment instead of concat;
    pandas flags this pattern as fragmentation-prone (PerformanceWarning).
    """
    X = X_in
    cols = X.columns.tolist()

    for switch in mapping:
        col = switch.get('col')
        mod = switch.get('mapping')

        codes = X[col].fillna(-2)
        for name, code in zip(mod.columns, mod.index[: len(mod.columns)], strict=True):
            X[name] = (codes == code).astype('int64')

        old_column_index = cols.index(col)
        cols[old_column_index : old_column_index + 1] = mod.columns

    return X.reindex(columns=cols)


def scenario(name: str, rows: int, n_cols: int, n_cats: int, seed: int = 0) -> None:
    """Measure fit, transform, and get_dummies-phase peaks for one shape."""
    print(f'--- {name}: {rows} rows x {n_cols} cols x {n_cats} categories ---')

    df = build_frame(rows, n_cols, n_cats, seed)
    enc = OneHotEncoder(cols=[f'c{i}' for i in range(n_cols)])

    fit_peak = peak_mb(enc.fit, df)
    df_input_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    # whole-call transform peak (input exists before the window)
    transform_peak = peak_mb(enc.transform, df)

    # get_dummies phase in isolation: ordinal codes built before the windows
    codes = enc.ordinal_encoder.transform(df)
    mapping = enc.mapping
    current_out = enc.get_dummies(codes)
    output_mb = current_out.memory_usage(deep=True).sum() / 1024 / 1024

    # each window sees only the call itself; entry frames are copied outside
    codes_for_current = codes.copy(deep=True)
    phase_current = peak_mb(enc.get_dummies, codes_for_current)
    codes_for_single = codes.copy(deep=True)
    phase_single_concat = peak_mb(dummies_single_concat, codes_for_single, mapping)
    codes_for_growth = codes.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        phase_bool_growth = peak_mb(dummies_bool_growth, codes_for_growth, mapping)

    # variant outputs must be identical to the current implementation's
    single_out = dummies_single_concat(codes, mapping)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        growth_out = dummies_bool_growth(codes, mapping)
    assert current_out.equals(single_out), 'single-concat output differs'
    assert current_out.equals(growth_out), 'bool-growth output differs'
    assert current_out.dtypes.equals(single_out.dtypes), 'single-concat dtypes differ'
    assert current_out.dtypes.equals(growth_out.dtypes), 'bool-growth dtypes differ'
    del current_out, single_out, growth_out

    print(f'  input frame:                  {df_input_mb:8.1f} MB')
    print(f'  fit peak:                     {fit_peak:8.1f} MB')
    print(f'  transform peak (whole call):  {transform_peak:8.1f} MB')
    print(f'  output frame:                 {output_mb:8.1f} MB')
    print(f'  get_dummies phase, current:   {phase_current:8.1f} MB')
    print(f'  get_dummies, single-concat:   {phase_single_concat:8.1f} MB')
    print(f'  get_dummies, bool-growth:     {phase_bool_growth:8.1f} MB')
    print()


def main() -> None:
    """Parse arguments and run the measured scenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--cols', type=int, default=10)
    parser.add_argument('--cats', type=int, default=20)
    parser.add_argument(
        '--scenario-b', action='store_true', help='also run the high-cardinality scenario'
    )
    args = parser.parse_args()

    scenario('A (reporter-like)', args.rows, args.cols, args.cats)
    if args.scenario_b:
        scenario('B (high-cardinality)', 20_000, 2, 500, seed=1)


if __name__ == '__main__':
    main()

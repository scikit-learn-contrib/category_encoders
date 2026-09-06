"""Memory repro for the #364 residual copies: WOE/OneHot fit and transform peaks.

Measures tracemalloc peaks of WOEEncoder and OneHotEncoder fit/transform on a
reporter-like synthetic frame. Used to record the before/after numbers for the
residual-copy cleanup (nested ``ordinal_encoder.transform`` wrapper copies and
WOE's double ordinal pass).

Usage: python3 benchmarks/repro-364.py [--rows 100000] [--cols 10] [--cats 20]

Default scenario (100k x 10 cols x 20 cats) matches the measurement in the
findings-memory-perf research: ~49 MB object input, WOE fit peak ~27 MB on
upstream master b2b8691 (post-#503: four stacked full-frame object copies per
fit). Run on the base commit and on this branch; the printed peaks are the
before/after numbers for the PR.

With ``--parity`` the script additionally prints SHA-256 digests of every
fitted mapping and transform output so two checkouts can be diffed for exact
behavioral parity.
"""

import argparse
import hashlib
import pickle
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from category_encoders import OneHotEncoder, WOEEncoder


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


def digest(obj) -> str:
    """Stable content digest of a frame, mapping, or arbitrary result."""
    return hashlib.sha256(pickle.dumps(obj, protocol=5)).hexdigest()


def woe_results(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit WOE outside any window and collect parity-relevant outputs."""
    enc = WOEEncoder(random_state=0, randomized=True).fit(X, y)
    mapping = {col: series.copy() for col, series in enc.mapping.items()}
    transformed = enc.transform(X)
    helper = enc.ordinal_encoder
    return {
        'woe_mapping': digest(mapping),
        'woe_transform': digest(transformed),
        'woe_ordinal_mapping': digest([dict(s) for s in helper.category_mapping]),
        'woe_ordinal_feature_names_out': digest(helper.get_feature_names_out()),
        'woe_feature_names_out': digest(enc.get_feature_names_out()),
    }


def onehot_results(X: pd.DataFrame) -> dict:
    """Fit OneHot outside any window and collect parity-relevant outputs."""
    enc = OneHotEncoder(use_cat_names=True).fit(X)
    return {
        'onehot_mapping': digest(enc.mapping),
        'onehot_transform': digest(enc.transform(X)),
        'onehot_ordinal_mapping': digest([dict(s) for s in enc.ordinal_encoder.category_mapping]),
        'onehot_feature_names_out': digest(enc.get_feature_names_out()),
    }


def main() -> None:
    """Measure fit/transform tracemalloc peaks and optionally print parity digests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--cols', type=int, default=10)
    parser.add_argument('--cats', type=int, default=20)
    parser.add_argument('--parity', action='store_true', help='print output digests')
    args = parser.parse_args()

    X = build_frame(args.rows, args.cols, args.cats)
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, size=len(X)), name='target')
    input_mb = X.memory_usage(deep=True).sum() / 1024 / 1024
    print(f'input frame: {args.rows} rows x {args.cols} cols x {args.cats} cats '
          f'= {input_mb:.1f} MB (object dtype)')

    woe = WOEEncoder(random_state=0, randomized=True)
    print(f'WOE    fit peak:            {peak_mb(woe.fit, X, y):8.1f} MB')
    woe_fitted = WOEEncoder(random_state=0, randomized=True).fit(X, y)
    print(f'WOE    transform peak:      {peak_mb(woe_fitted.transform, X):8.1f} MB')
    woe_ft = WOEEncoder(random_state=0, randomized=True)
    print(f'WOE    fit_transform peak:  {peak_mb(woe_ft.fit_transform, X, y):8.1f} MB')

    onehot = OneHotEncoder()
    print(f'OneHot fit peak:            {peak_mb(onehot.fit, X):8.1f} MB')
    onehot_fitted = OneHotEncoder().fit(X)
    print(f'OneHot transform peak:      {peak_mb(onehot_fitted.transform, X):8.1f} MB')

    if args.parity:
        for key, value in {**woe_results(X, y), **onehot_results(X)}.items():
            print(f'{key}: {value}')


if __name__ == '__main__':
    main()

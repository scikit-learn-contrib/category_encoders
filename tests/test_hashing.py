"""Tests for the HashingEncoder."""
import hashlib
import math
import multiprocessing
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest


class TestHashingEncoder(TestCase):
    """Tests for the HashingEncoder."""

    def test_must_not_reset_index(self):
        """Test that the HashingEncoder does not reset the index."""
        columns = ['column1', 'column2', 'column3', 'column4']
        df = pd.DataFrame([[i, i, i, i] for i in range(10)], columns=columns)
        df = df.iloc[2:8, :]
        target_columns = ['column1', 'column2', 'column3']

        single_process_encoder = encoders.HashingEncoder(max_process=1, cols=target_columns)
        single_process_encoder.fit(df, None)
        df_encoded_single_process = single_process_encoder.transform(df)
        assert_index_equal(df.index, df_encoded_single_process.index)
        self.assertEqual(df.shape[0],
                         pd.concat([df, df_encoded_single_process], axis=1).shape[0])

        multi_process_encoder = encoders.HashingEncoder(cols=target_columns)
        multi_process_encoder.fit(df, None)
        df_encoded_multi_process = multi_process_encoder.transform(df)
        assert_index_equal(df.index, df_encoded_multi_process.index)
        self.assertEqual(df.shape[0] , pd.concat([df, df_encoded_multi_process], axis=1).shape[0])

        assert_frame_equal(df_encoded_single_process, df_encoded_multi_process)

    def test_transform_works_with_single_row_df(self):
        """Test that the HashingEncoder works with a single row DataFrame."""
        columns = ['column1', 'column2', 'column3', 'column4']
        df = pd.DataFrame([[i, i, i, i] for i in range(10)], columns=columns)
        df = df.iloc[2:8, :]
        target_columns = ['column1', 'column2', 'column3']

        multi_process_encoder = encoders.HashingEncoder(cols=target_columns)
        multi_process_encoder.fit(df, None)
        df_encoded_multi_process = multi_process_encoder.transform(df.sample(1))

        self.assertEqual(
            multi_process_encoder.n_components + len(list(set(columns) - set(target_columns))),
            df_encoded_multi_process.shape[1]
        )

    def test_simple_example(self):
        """Test the HashingEncoder with a simple example."""
        df = pd.DataFrame(
            {
                'strings': ['aaaa', 'bbbb', 'cccc'],
                'more_strings': ['aaaa', 'dddd', 'eeee'],
            }
        )
        encoder = encoders.HashingEncoder(n_components=4, max_process=2)
        encoder.fit(df)
        expected_df = pd.DataFrame(
            {'col_0': [0, 1, 1], 'col_1': [2, 0, 1], 'col_2': [0, 1, 0], 'col_3': [0, 0, 0]}
        )
        pd.testing.assert_frame_equal(encoder.transform(df), expected_df)


class TestHashingHardening(TestCase):
    """Hardening tests for HashingEncoder mutation survivors."""

    @staticmethod
    def _bucket(value, n_components=8, method='md5'):
        """Independent hashlib reference for the bucket a value lands in."""
        digest = hashlib.new(method, str(value).encode('utf-8'))
        return int(digest.hexdigest(), 16) % n_components

    def test_duplicate_values_within_a_row_accumulate(self):
        """Two identical values in one row increment the same bucket twice.

        Mutant category_encoders/hashing.py::hash_chunk mutmut_34
        (counts += 1 -> = 1) collapses duplicates to a single hit.
        """
        df = pd.DataFrame({'a': ['x', 'x'], 'b': ['x', None]})
        encoder = encoders.HashingEncoder(n_components=8)
        encoder.fit(df)
        out = encoder.transform(df)
        bucket = self._bucket('x')
        self.assertEqual(out.iloc[0, bucket], 2)
        self.assertEqual(out.iloc[0].sum(), 2)

    def test_none_values_are_skipped(self):
        """None entries are not hashed; only real values hit buckets.

        Mutant category_encoders/hashing.py::hash_chunk mutmut_16
        (if val is not None -> if val is None) hashes None instead of the
        present values, leaving the value's bucket cold.
        """
        df = pd.DataFrame({'a': ['x', 'x'], 'b': [None, None]})
        encoder = encoders.HashingEncoder(n_components=8)
        encoder.fit(df)
        out = encoder.transform(df)
        bucket = self._bucket('x')
        self.assertEqual(out.iloc[0, bucket], 1)
        self.assertEqual(out.iloc[1, bucket], 1)
        self.assertEqual(out.iloc[0].sum(), 1)

    def test_hash_method_is_respected(self):
        """The configured hash_method must drive the hashing.

        Mutant category_encoders/hashing.py::_transform mutmut_17 drops the
        hashing_method keyword, silently falling back to the md5 default.
        """
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        md5_out = (
            encoders.HashingEncoder(n_components=8, hash_method='md5')
            .fit(df)
            .transform(df)
        )
        sha_out = (
            encoders.HashingEncoder(n_components=8, hash_method='sha256')
            .fit(df)
            .transform(df)
        )
        self.assertFalse(np.array_equal(md5_out.to_numpy(), sha_out.to_numpy()))

    def test_default_construction_attributes(self):
        """Documented constructor defaults survive mutation.

        Mutants dropping super().__init__ keywords (handle_unknown /
        handle_missing) or flipping defaults (n_components, hash_method,
        verbose) are visible on these attributes.
        """
        encoder = encoders.HashingEncoder()
        self.assertEqual(encoder.n_components, 8)
        self.assertEqual(encoder.hash_method, 'md5')
        self.assertEqual(encoder.verbose, 0)
        self.assertEqual(encoder.handle_unknown, 'does not apply')
        self.assertEqual(encoder.handle_missing, 'does not apply')

    def test_max_process_ladder_clamps_auto_selection(self):
        """max_process=0 auto-selects a worker count clamped to [1, 128].

        Covers the auto-selection ladder in __init__ (the non-Windows branch
        with its lower and upper clamps) and passes an explicit value through.
        """
        expected = min(max(int(math.ceil(multiprocessing.cpu_count() / 2)), 1), 128)
        self.assertEqual(expected, encoders.HashingEncoder().max_process)
        self.assertEqual(1, encoders.HashingEncoder(max_process=1).max_process)

    def test_parallel_path_matches_single_process_result(self):
        """The ProcessPoolExecutor path hashes identically to the single-process path."""
        df = pd.DataFrame({'c': ['a', 'b', 'c'] * 6, 'd': ['x', 'y'] * 9})
        single = encoders.HashingEncoder(max_process=1, n_components=4).fit_transform(df)
        parallel = encoders.HashingEncoder(max_process=2, n_components=4).fit_transform(df)
        assert_frame_equal(single, parallel)

    def test_max_process_windows_and_clamp_ladder(self):
        """Windows picks 1 worker with spawn; the CPU-derived count clamps to [1, 128].

        Covers the platform branches of the __init__ ladder that a Linux host
        cannot reach naturally.
        """
        from unittest.mock import patch

        with (
            patch('category_encoders.hashing.platform.system', return_value='Windows'),
            patch('category_encoders.hashing.multiprocessing.cpu_count', return_value=8),
        ):
            windows_encoder = encoders.HashingEncoder()
        self.assertEqual(1, windows_encoder.max_process)
        self.assertEqual('spawn', windows_encoder.process_creation_method)

        with (
            patch('category_encoders.hashing.platform.system', return_value='Linux'),
            patch('category_encoders.hashing.multiprocessing.cpu_count', return_value=0),
        ):
            self.assertEqual(1, encoders.HashingEncoder().max_process)

        with (
            patch('category_encoders.hashing.platform.system', return_value='Linux'),
            patch('category_encoders.hashing.multiprocessing.cpu_count', return_value=300),
        ):
            self.assertEqual(128, encoders.HashingEncoder().max_process)

    def test_transform_before_fit_raises(self):
        """Unfitted transform raises the documented NotFittedError contract."""
        df = pd.DataFrame({'c': ['a', 'b']})
        with self.assertRaisesRegex(ValueError, 'not fitted yet'):
            encoders.HashingEncoder().transform(df)

    def test_transform_with_no_cols_is_identity(self):
        """cols=[] short-circuits transform to the (converted) input frame."""
        df = pd.DataFrame({'c': ['a', 'b']})
        encoder = encoders.HashingEncoder(cols=[])
        encoder.fit(df)
        assert_frame_equal(df, encoder.transform(df))

    def test_hashing_trick_rejects_unknown_method(self):
        """hashing_trick rejects a hashing method hashlib does not offer."""
        df = pd.DataFrame({'c': ['a', 'b']})
        with self.assertRaisesRegex(ValueError, 'not available'):
            encoders.HashingEncoder().hashing_trick(df, hashing_method='not-a-hash')

    def test_hashing_trick_without_copy_hashes_in_place(self):
        """make_copy=False hashes the frame in place and returns the same object."""
        df = pd.DataFrame({'c': ['a', 'b']})
        result = encoders.HashingEncoder().hashing_trick(df.copy(), make_copy=False)
        # With cols=None every column is hashed away and replaced by hashed columns.
        self.assertEqual([], [col for col in result.columns if col == 'c'])

    def test_hashing_trick_hashes_all_columns_by_default(self):
        """hashing_trick with cols=None hashes every column.

        Mutant category_encoders/hashing.py::hashing_trick mutmut_15
        (cols = X.columns -> cols = None) selects no columns.
        """
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q']})
        trick = encoders.HashingEncoder().hashing_trick(df, N=3)
        self.assertEqual(trick.shape, (2, 3))
        self.assertEqual(trick.iloc[0].sum(), 2)


def test_default_constructor_attributes():
    enc = encoders.HashingEncoder()
    assert enc.max_sample == 0
    assert enc.n_components == 8
    assert enc.hash_method == 'md5'
    expected_max_process = min(max(int(math.ceil(multiprocessing.cpu_count() / 2)), 1), 128)
    assert enc.max_process == expected_max_process


def test_pooling_params_are_stored():
    enc = encoders.HashingEncoder(
        min_group_size=5, min_group_name='rare', combine_min_nan_groups=True
    )
    assert enc.min_group_size == 5
    assert enc.min_group_name == 'rare'
    assert enc.combine_min_nan_groups is True


def test_max_process_above_limit_recomputes_from_cpu_count():
    # 128 is outside the accepted range(1, 128), so the encoder must fall back
    # to half the logical CPU count rather than accepting the passed value.
    enc = encoders.HashingEncoder(max_process=128)
    expected = min(max(int(math.ceil(multiprocessing.cpu_count() / 2)), 1), 128)
    assert enc.max_process == expected


def test_transform_before_fit_exact_message():
    enc = encoders.HashingEncoder(cols=['a'])
    X = pd.DataFrame({'a': ['x', 'y']})
    with pytest.raises(
        ValueError,
        match=r'^Must train encoder before it can be used to transform data\.$',
    ):
        enc._transform(X)


def test_output_dtype_is_integer():
    X = pd.DataFrame({'a': ['x', 'y', 'z']})
    out = encoders.HashingEncoder(cols=['a']).fit_transform(X)
    assert all(np.issubdtype(dt, np.integer) for dt in out.dtypes)


def test_hash_method_changes_encoding():
    X = pd.DataFrame({'a': ['hello', 'world', 'foo', 'bar']})
    md5_out = encoders.HashingEncoder(hash_method='md5').fit_transform(X)
    sha_out = encoders.HashingEncoder(hash_method='sha256').fit_transform(X)
    assert not md5_out.equals(sha_out)


def test_parallel_matches_serial_output():
    X = pd.DataFrame({'a': ['x', 'y', 'z', 'x'] * 10})
    parallel = encoders.HashingEncoder(max_process=2).fit_transform(X)
    serial = encoders.HashingEncoder(max_process=1).fit_transform(X)
    assert_frame_equal(parallel, serial)

"""Tests for the HashingEncoder."""
import hashlib
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal


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

    def test_hashing_trick_hashes_all_columns_by_default(self):
        """hashing_trick with cols=None hashes every column.

        Mutant category_encoders/hashing.py::hashing_trick mutmut_15
        (cols = X.columns -> cols = None) selects no columns.
        """
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q']})
        trick = encoders.HashingEncoder().hashing_trick(df, N=3)
        self.assertEqual(trick.shape, (2, 3))
        self.assertEqual(trick.iloc[0].sum(), 2)

from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal

import category_encoders as encoders


class TestHashingEncoder(TestCase):
    def test_must_not_reset_index(self):
        columns = ['column1', 'column2', 'column3', 'column4']
        df = pd.DataFrame([[i, i, i, i] for i in range(10)], columns=columns)
        df = df.iloc[2:8, :]
        target_columns = ['column1', 'column2', 'column3']

        single_process_encoder = encoders.HashingEncoder(max_process=1, cols=target_columns)
        single_process_encoder.fit(df, None)
        df_encoded_single_process = single_process_encoder.transform(df)
        assert_index_equal(df.index, df_encoded_single_process.index)
        assert df.shape[0] == pd.concat([df, df_encoded_single_process], axis=1).shape[0]

        multi_process_encoder = encoders.HashingEncoder(cols=target_columns)
        multi_process_encoder.fit(df, None)
        df_encoded_multi_process = multi_process_encoder.transform(df)
        assert_index_equal(df.index, df_encoded_multi_process.index)
        assert df.shape[0] == pd.concat([df, df_encoded_multi_process], axis=1).shape[0]

        assert_frame_equal(df_encoded_single_process, df_encoded_multi_process)

"""Round-trip regressions for original columns sharing an encoded prefix."""

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    'encoder_cls,kwargs',
    [
        (encoders.BaseNEncoder, {'base': 1}),
        (encoders.BaseNEncoder, {'base': 2}),
        (encoders.BaseNEncoder, {'base': 3}),
        (encoders.BinaryEncoder, {}),
        (encoders.GrayEncoder, {}),
    ],
)
@pytest.mark.parametrize(
    'neighbor,encode_neighbor',
    [('value', False), ('cat_9', False), ('cat_9_suffix', False), ('cat_9', True)],
)
@pytest.mark.parametrize('return_df', [True, False])
def test_inverse_transform_preserves_prefixed_columns(
    encoder_cls, kwargs, neighbor, encode_neighbor, return_df
):
    """Decode only fitted digits and preserve nearby original columns."""
    data = pd.DataFrame(
        {
            'left': [1, 2, 3, 4],
            'cat': ['a', 'b', 'c', 'a'],
            neighbor: ['b', 'c', 'c', 'b'] if encode_neighbor else [5, 6, 7, 8],
            'right': [4, 3, 2, 1],
        },
        index=[20, 21, 22, 23],
    )
    cols = ['cat', neighbor] if encode_neighbor else ['cat']
    encoder = encoder_cls(cols=cols, return_df=return_df, **kwargs)
    encoded = encoder.fit_transform(data)
    encoded_before = encoded.copy()

    restored = encoder.inverse_transform(encoded)

    if return_df:
        pd.testing.assert_frame_equal(restored, data)
        pd.testing.assert_frame_equal(encoded, encoded_before)
    else:
        np.testing.assert_array_equal(restored, data.to_numpy())
        np.testing.assert_array_equal(encoded, encoded_before)

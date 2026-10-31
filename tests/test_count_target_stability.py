"""Numerical stability of smoothed categorical log-evidence."""

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from category_encoders import CountTargetEncoder
from numpy.testing import assert_allclose


@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('smoothing', [0.5, 10.0, 1000.0])
def test_pure_category_log_evidence(n_classes, smoothing):
    """Finite smoothing keeps evidence finite even for large pure groups."""
    size = 400
    y = np.repeat(np.arange(n_classes), size)
    X = pd.DataFrame({'cat': y.astype(str)})
    actual = CountTargetEncoder(smoothing=smoothing).fit_transform(X, y).to_numpy()

    with localcontext() as context:
        context.prec = 100
        prior = Decimal(1) / n_classes
        z = (Decimal(size) - 20) / Decimal(str(smoothing))
        exp_z = z.exp()
        expected = []
        for label in y:
            evidence = [
                ((prior + (exp_z if label == cls else 0)) / (1 + exp_z)).ln() - prior.ln()
                for cls in range(n_classes)
            ]
            expected.append(
                [float(evidence[1] - evidence[0])]
                if n_classes == 2
                else [float(value) for value in evidence]
            )
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

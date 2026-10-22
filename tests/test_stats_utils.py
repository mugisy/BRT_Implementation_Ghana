import numpy as np
import pandas as pd

from brt.stats_utils import (compute_vif, cramers_v, holm_correct,
                             nadeau_bengio_ttest)


def test_holm_matches_manual():
    adj = holm_correct([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_cramers_v_perfect_association():
    x = pd.Series([0, 1] * 200)
    v, p = cramers_v(x, x, bias_correction=False)
    assert v > 0.99 and p < 1e-10


def test_nadeau_bengio_wider_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.01, 50)
    _, p_nb, *_ = nadeau_bengio_ttest(d, 0.25)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert p_nb > 0 and abs(naive_t) > 0


def test_vif_independent_columns_near_one():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    assert (compute_vif(X) < 1.2).all()
"""Statistical helpers shared across the pipeline."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------
# Association measures
# ---------------------------------------------------------------------
def cramers_v(x, y, bias_correction: bool = True):
    """Cramér's V and chi-square p-value.

    bias_correction=True uses the Bergsma (2013) correction.
    """
    ct = pd.crosstab(x, y)
    r, k = ct.shape
    if min(r, k) < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    if not bias_correction:
        return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))), p
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan, p
    return float(np.sqrt(phi2corr / denom)), p


# ---------------------------------------------------------------------
# Logistic regression + likelihood ratio tests
# ---------------------------------------------------------------------
def fit_logit(X, y, standardize: bool = True, check_convergence: bool = True,
              maxiter: int = 200):
    """Fit a statsmodels Logit.

    standardize=True  -> z-scored predictors (comparable LR tests)
    standardize=False -> raw scale, keeps column names (odds ratios)
    """
    X = pd.DataFrame(X).astype(float)
    if standardize:
        Xc = sm.add_constant(StandardScaler().fit_transform(X),
                             has_constant="add")
    else:
        Xc = sm.add_constant(X, has_constant="add")
    m = sm.Logit(y, Xc).fit(method="newton", maxiter=maxiter, disp=0)
    if check_convergence and not m.mle_retvals.get("converged", True):
        raise RuntimeError("Logit did not converge.")
    return m


def lr_test(m_full, m_restricted, df_diff=None):
    """Likelihood ratio test -> (LR, df, p)."""
    lr = 2 * (m_full.llf - m_restricted.llf)
    if df_diff is None:
        df_diff = m_full.df_model - m_restricted.df_model
    return lr, df_diff, stats.chi2.sf(lr, df_diff)


# ---------------------------------------------------------------------
# Multicollinearity
# ---------------------------------------------------------------------
def compute_vif(X: pd.DataFrame) -> pd.Series:
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    vals = [variance_inflation_factor(Xc.values, i)
            for i in range(1, Xc.shape[1])]
    return pd.Series(vals, index=X.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------
# Resampling inference & multiplicity
# ---------------------------------------------------------------------
def nadeau_bengio_ttest(diffs, test_train_ratio: float):
    """Corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert & Frank
    2004). Returns (t, p, mean, se, k)."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    k = len(d)
    se = np.sqrt(d.var(ddof=1) * (1.0 / k + test_train_ratio))
    t = d.mean() / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), k - 1) if np.isfinite(t) else np.nan
    return t, p, d.mean(), se, k


def nb_confint(diffs, test_train_ratio: float, alpha: float = 0.05):
    _, _, mean, se, k = nadeau_bengio_ttest(diffs, test_train_ratio)
    tcrit = stats.t.ppf(1 - alpha / 2, k - 1)
    return mean - tcrit * se, mean + tcrit * se


def holm_correct(pvals):
    """Holm step-down adjusted p-values (returned in input order)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running_max = max(running_max, min((m - rank) * p[idx], 1.0))
        adj[idx] = running_max
    return adj


def bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    return np.minimum(p * len(p), 1.0)
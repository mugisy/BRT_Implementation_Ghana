"""STEP 4 — Consolidated feature-justification tests.

Test 1  — joint + per-feature LRT for the 'non-significant' block
Test 1b — targeted 2-df test for post-hoc candidates (exploratory)
Test 2  — repeated-CV predictive gain (LR, RF, GB)
Test 3  — permutation importance (held-out set)
Test 4  — interaction screening (weak x strong), Bonferroni-corrected
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data import build_feature_sets, design_matrices, get_target
from .io_utils import ensure_dir, save_table, section
from .stats_utils import (bonferroni, fit_logit, lr_test, nadeau_bengio_ttest)


# ---------------------------------------------------------------------
def test1_lrt(X_reduced, X_full, y, non_sig, df, sig, candidates=config.CANDIDATES):
    section("TEST 1 — Joint LRT for excluded block")
    m_red, m_full = fit_logit(X_reduced, y), fit_logit(X_full, y)
    lr, df_, p = lr_test(m_full, m_red)
    print(f"  LR = {lr:.3f}, df = {df_}, p = {p:.4f}")

    n = len(non_sig)
    print(f"\n  Per-feature LRTs (Bonferroni alpha = {0.05 / n:.4f}):")
    rows = []
    for feat in non_sig:
        lr_i, _, p_i = lr_test(m_full, fit_logit(X_full.drop(columns=[feat]), y), 1)
        flag = " **" if p_i < 0.05 / n else (" *" if p_i < 0.05 else "")
        print(f"    {feat:<22} LR = {lr_i:6.3f}  p = {p_i:.4f}{flag}")
        rows.append({"feature": feat, "LR": lr_i, "p": p_i})

    cands = [c for c in candidates if c in df.columns]
    m_c = fit_logit(df[sig + cands].astype(float), y)
    lr_c, df_c, p_c = lr_test(m_c, m_red)
    print(f"\n  TEST 1b — {cands}: LR = {lr_c:.3f}, df = {df_c}, "
          f"p = {p_c:.4f}  (exploratory)")
    return {"joint": (lr, df_, p), "per_feature": pd.DataFrame(rows),
            "candidates": (lr_c, df_c, p_c)}


def default_models(rng=config.RNG):
    return [("Logistic Regression", LogisticRegression(max_iter=2000)),
            ("Random Forest", RandomForestClassifier(n_estimators=500,
                                                     random_state=rng)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=rng))]


def test2_cv_gain(X_reduced, X_full, y, n_splits=config.N_SPLITS,
                  n_repeats=config.N_REPEATS, rng=config.RNG):
    """Paired comparison on identical folds. Reports both the naive paired
    t-test (as in the original notebook) and the Nadeau–Bengio corrected
    test, which accounts for overlapping training sets."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=rng)
    ratio = 1.0 / (n_splits - 1)
    rows = []
    for name, model in default_models(rng):
        section(f"TEST 2 — {name} ({cv.get_n_splits()} resamples)")
        pipe = Pipeline([("scale", StandardScaler()), ("clf", model)])
        for scoring, label in [("roc_auc", "AUC"), ("average_precision", "AP ")]:
            s_red = cross_val_score(pipe, X_reduced, y, cv=cv, scoring=scoring)
            s_full = cross_val_score(pipe, X_full, y, cv=cv, scoring=scoring)
            t, p = stats.ttest_rel(s_full, s_red)
            t_nb, p_nb, *_ = nadeau_bengio_ttest(s_full - s_red, ratio)
            print(f"  [{label}] Reduced {s_red.mean():.4f}±{s_red.std():.4f}  "
                  f"Full {s_full.mean():.4f}±{s_full.std():.4f}  "
                  f"t = {t:.3f}, p = {p:.4f} | t_NB = {t_nb:.3f}, "
                  f"p_NB = {p_nb:.4f}")
            rows.append({"model": name, "metric": label.strip(),
                         "reduced": s_red.mean(), "full": s_full.mean(),
                         "t_naive": t, "p_naive": p, "t_NB": t_nb, "p_NB": p_nb})
    return pd.DataFrame(rows)


def test3_permutation_importance(X_full, y, significant, model=None,
                                 rng=config.RNG, n_repeats=50):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.3, stratify=y, random_state=rng)
    if model is None:
        model = RandomForestClassifier(n_estimators=500, random_state=rng)
    model.fit(X_tr, y_tr)
    perm = permutation_importance(model, X_te, y_te, n_repeats=n_repeats,
                                  scoring="roc_auc", random_state=rng)
    imp = pd.DataFrame({
        "feature": X_full.columns, "importance": perm.importances_mean,
        "std": perm.importances_std,
        "group": ["sig" if f in significant else "NON-SIG"
                  for f in X_full.columns],
    }).sort_values("importance", ascending=False)
    print(imp.to_string(index=False))
    print("  (Rank-block importances are shared among the remaining rank "
          "variables — interpret as a block.)")
    return imp


def test4_interaction_screen(X_full, y, weak, strong):
    Xs = pd.DataFrame(StandardScaler().fit_transform(X_full),
                      columns=X_full.columns)
    m_base = sm.Logit(y, sm.add_constant(Xs)).fit(method="newton",
                                                  maxiter=200, disp=0)
    rows = []
    for w in weak:
        for s in strong:
            X_int = Xs.copy()
            X_int["interaction"] = X_int[w] * X_int[s]
            X_int_c = sm.add_constant(X_int)
            if np.linalg.cond(X_int_c) > 1e8:
                print(f"    {w} x {s}: ill-conditioned — skipped")
                continue
            try:
                m_int = sm.Logit(y, X_int_c).fit(method="newton",
                                                 maxiter=200, disp=0)
            except np.linalg.LinAlgError:
                print(f"    {w} x {s}: singular matrix — skipped")
                continue
            lr, _, p = lr_test(m_int, m_base, 1)
            rows.append((w, s, lr, p))
    res = pd.DataFrame(rows, columns=["weak", "strong", "LR", "p"])
    res["p_bonf"] = bonferroni(res["p"].values)
    sig = res[res["p_bonf"] < 0.05].sort_values("p_bonf")
    print(f"  ({len(res)} interactions tested; Bonferroni threshold = "
          f"{0.05 / max(len(res), 1):.4f})\n")
    print(sig.to_string(index=False) if len(sig)
          else res.nsmallest(5, "p").to_string(index=False))
    return res.sort_values("p")


# ---------------------------------------------------------------------
def run_consolidated_tests(df, output_dir, significant=config.SIGNIFICANT,
                           n_repeats=config.N_REPEATS):
    out = ensure_dir(output_dir / "consolidated")
    y = get_target(df)
    fs = build_feature_sets(df, significant=significant)
    X_reduced, X_full = design_matrices(df, fs)
    print(f"n = {len(y)}, positives = {y.mean():.1%} -> metrics: AUC & AP, "
          "never accuracy.")

    t1 = test1_lrt(X_reduced, X_full, y, fs.non_significant, df, fs.significant)
    save_table(t1["per_feature"], out / "test1_per_feature_lrt.csv", index=False)

    t2 = test2_cv_gain(X_reduced, X_full, y, n_repeats=n_repeats)
    save_table(t2, out / "test2_cv_gain.csv", index=False)

    section("TEST 3 — Permutation importance (Random Forest, AUC drop)")
    t3 = test3_permutation_importance(X_full, y, fs.significant)
    save_table(t3, out / "test3_permutation_importance.csv", index=False)

    section("TEST 4 — Interaction screening (weak x strong)")
    t4 = test4_interaction_screen(X_full, y, fs.non_significant, fs.significant)
    save_table(t4, out / "test4_interactions.csv", index=False)
    return {"test1": t1, "test2": t2, "test3": t3, "test4": t4}
"""STEP 3 — Explanatory models A and B ("continue the analysis").

Odds ratios, block LR tests, mediation contrast (A -> B), CV AUC, and a
theory-driven interaction screen (Holm-corrected).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .feature_engineering import DesignSpec, build_design_matrices
from .io_utils import ensure_dir, save_table, section
from .stats_utils import fit_logit, holm_correct, lr_test

MEDIATION_FOCUS = ["income", "car_available", "hse_vehicle_ownership"]
INTERACTION_CANDIDATES = [
    ("gender", "travel_assault"),     # gendered safety concerns
    ("income", "car_available"),      # car as luxury vs default
    ("education_level", "income"),
    ("activity_grp", "income"),       # dropped automatically (not in X_A)
    ("age_ord", "mobility_issues"),   # only if age is ordinal-coded
]


def odds_ratio_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    return pd.DataFrame({
        "OR": np.exp(model.params), "CI_lo": np.exp(ci[0]),
        "CI_hi": np.exp(ci[1]), "p": model.pvalues,
    }).drop("const").sort_values("p")


def block_lr_tests(X, y, m_full, blocks: dict) -> pd.DataFrame:
    rows = []
    for label, cols in blocks.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        m_r = fit_logit(X.drop(columns=cols), y, standardize=False,
                        check_convergence=False)
        lr, df_, p = lr_test(m_full, m_r, len(cols))
        rows.append({"block": label, "df": df_, "LR": lr, "p": p})
    return pd.DataFrame(rows)


def mediation_contrast(m_A, m_B, X_A, X_B, focus=MEDIATION_FOCUS):
    shared = [c for c in X_A.columns if c in X_B.columns]
    c = pd.DataFrame({"beta_A": m_A.params[shared], "beta_B": m_B.params[shared]})
    c["attenuation_%"] = 100 * (1 - c["beta_B"] / c["beta_A"])
    return c.loc[[f for f in focus if f in c.index]]


def cv_auc(designs: dict, y, rng=config.RNG) -> pd.DataFrame:
    cv = StratifiedKFold(5, shuffle=True, random_state=rng)
    rows = []
    for name, X in designs.items():
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
        auc = cross_val_score(pipe, X.astype(float), y, cv=cv,
                              scoring="roc_auc")
        rows.append({"model": name, "auc_mean": auc.mean(),
                     "auc_std": auc.std()})
    return pd.DataFrame(rows)


def interaction_screen(X, y, m_base, candidates=INTERACTION_CANDIDATES):
    candidates = [c for c in candidates if all(v in X.columns for v in c)]
    rows = []
    for a, b in candidates:
        Xi = X.copy()
        Xi[f"{a}_x_{b}"] = Xi[a] * Xi[b]
        m_i = fit_logit(Xi, y, standardize=False, check_convergence=False)
        lr, _, p = lr_test(m_i, m_base, 1)
        rows.append({"interaction": f"{a} x {b}", "LR": lr, "p_raw": p})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm_correct(res["p_raw"].values)
        res = res.sort_values("p_raw")
    return res


def run_explanatory_analysis(df: pd.DataFrame, output_dir,
                             spec: DesignSpec = None) -> dict:
    out = ensure_dir(output_dir / "explanatory")
    if spec is None:
        spec = build_design_matrices(df, verbose=False)
    X_A, X_B, y = spec.X_A, spec.X_B, spec.y

    section("STEP 3 — Explanatory models")
    m_A = fit_logit(X_A, y, standardize=False)
    m_B = fit_logit(X_B, y, standardize=False)
    print(f"  Model A: LL = {m_A.llf:.1f}, pseudo-R2 = {m_A.prsquared:.3f}")
    print(f"  Model B: LL = {m_B.llf:.1f}, pseudo-R2 = {m_B.prsquared:.3f}")

    or_tab = odds_ratio_table(m_A)
    print("\n  Model A odds ratios:\n", or_tab.round(3).to_string())
    save_table(or_tab, out / "model_A_odds_ratios.csv")
    save_table(odds_ratio_table(m_B), out / "model_B_odds_ratios.csv")

    blocks = {
        "rank_block": spec.rank_feats,
        "phone_block": spec.phone_feats,
        "experience_block": spec.exp_feat,
        "age_block": spec.age_cols,
        "private_mobility": ["driving_license", "car_available",
                             "hse_vehicle_ownership"],
        "activity_block": spec.act_cols,
    }
    blk = block_lr_tests(X_A, y, m_A, blocks)
    print("\n  Block LR tests (Model A):\n", blk.round(4).to_string(index=False))
    save_table(blk, out / "block_lr_tests.csv", index=False)

    med = mediation_contrast(m_A, m_B, X_A, X_B)
    print("\n  Mediation contrast (attenuation = effect through mode choice):")
    print(med.round(3).to_string())
    save_table(med, out / "mediation_contrast.csv")

    auc = cv_auc({"A": X_A, "B": X_B}, y)
    for _, r in auc.iterrows():
        print(f"  Model {r['model']}: CV AUC = {r['auc_mean']:.3f} "
              f"+/- {r['auc_std']:.3f}")
    save_table(auc, out / "cv_auc.csv", index=False)

    ints = interaction_screen(X_A, y, m_A)
    print("\n  Interaction screen:\n", ints.round(4).to_string(index=False))
    save_table(ints, out / "interaction_screen.csv", index=False)

    return {"m_A": m_A, "m_B": m_B, "odds_ratios": or_tab, "blocks": blk,
            "mediation": med, "cv_auc": auc, "interactions": ints}
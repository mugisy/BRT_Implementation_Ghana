"""STEP 2 — Feature combination using Cramér's V and VIF.

Builds two design matrices:
  Model A (explanatory): mode choice excluded (mediator; V=0.64 with license)
  Model B (predictive) : mode included, driving_license dropped
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from . import config
from .data import check_rank_block, get_target
from .io_utils import ensure_dir, save_table, section
from .stats_utils import compute_vif, cramers_v, fit_logit, lr_test

ACTIVITY_MAP = {0: "formal", 1: "formal",
                2: "informal", 3: "informal",
                4: "not_working", 5: "not_working", 8: "not_working",
                6: "student", 7: "student"}

MODE_MAP_3 = {0: "private_motor", 3: "private_motor",
              1: "non_motorized", 2: "non_motorized",
              **{k: "shared_public" for k in (4, 5, 6, 7, 8, 9, 10)}}

MODE_MAP_4 = {**MODE_MAP_3, 7: "for_hire", 8: "for_hire"}

FOR_HIRE_CODES = (7, 8)
FOR_HIRE_MIN_N = 150
PHONE_UNIVERSAL_THRESHOLD = 0.95
SKEW_THRESHOLD = 1.0

REDUNDANCY_PAIRS = [("driving_license", "owns_vehicle"),
                    ("driving_license", "mode_grp_3"),
                    ("age_grp_c", "activity_grp"),
                    ("income", "education_level"),
                    ("income", "activity_grp")]


@dataclass
class DesignSpec:
    X_A: pd.DataFrame
    X_B: pd.DataFrame
    y: pd.Series
    exp_feat: List[str]
    phone_feats: List[str]
    rank_feats: List[str]
    age_cols: List[str]
    act_cols: List[str]
    mode_cols: List[str]
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ordinal age (all 7 levels) + clipped version for redundancy checks
    out["age_ord"] = out["age_group"].astype(float)
    out["age_sq"] = out["age_ord"] ** 2
    out["age_grp_c"] = out["age_group"].clip(lower=1, upper=5)
    # Phone (0=No, 1=Yes+Internet, 2=Yes-no Internet)
    out["has_phone"] = (out["phone_ownership"] != 0).astype(int)
    out["has_internet"] = (out["phone_ownership"] == 1).astype(int)
    # Activity: 9 -> 4 groups
    out["activity_grp"] = out["current_activity"].map(ACTIVITY_MAP)
    # Mode: 11 -> 3 (or 4 if for-hire is large enough)
    out["mode_grp_3"] = out["transport_to_work"].map(MODE_MAP_3)
    for_hire_n = out["transport_to_work"].isin(FOR_HIRE_CODES).sum()
    out["mode_grp"] = (out["transport_to_work"].map(MODE_MAP_4)
                       if for_hire_n >= FOR_HIRE_MIN_N else out["mode_grp_3"])
    # Private mobility access
    out["owns_vehicle"] = (out["hse_vehicle_ownership"] > 0).astype(int)
    out["car_available"] = (out["driving_license"]
                            * out["owns_vehicle"]).astype(int)
    # Transport expenditure
    out["log_transport_exp"] = np.log1p(out["monthly_transport_exp"])
    return out


def phone_features(df) -> List[str]:
    if df["has_phone"].mean() > PHONE_UNIVERSAL_THRESHOLD:
        print("  Phone ownership near-universal; keeping only has_internet.")
        return ["has_internet"]
    return ["has_phone", "has_internet"]


def expenditure_feature(df) -> List[str]:
    return (["log_transport_exp"]
            if df["monthly_transport_exp"].skew() > SKEW_THRESHOLD
            else ["monthly_transport_exp"])


def age_encoding(df, y):
    """Compare ordinal, quadratic and full-dummy age encodings."""
    age_dum = pd.get_dummies(df["age_group"], prefix="age",
                             drop_first=True).astype(int)
    m_ord = fit_logit(df[["age_ord"]], y, check_convergence=False)
    m_quad = fit_logit(df[["age_ord", "age_sq"]], y, check_convergence=False)
    m_dum = fit_logit(age_dum, y, check_convergence=False)
    lr_lin, df_lin, p_lin = lr_test(m_dum, m_ord)
    lr_q, df_q, p_q = lr_test(m_dum, m_quad)
    print(f"  Age linearity test : LR = {lr_lin:.2f}, df = {df_lin}, "
          f"p = {p_lin:.4f}")
    print(f"  Quadratic vs dummy : LR = {lr_q:.2f}, df = {df_q}, "
          f"p = {p_q:.4f}")
    if p_q < config.ALPHA:
        print("  -> Quadratic insufficient: using full age dummies")
        return age_dum, list(age_dum.columns), {"p_linear": p_lin,
                                                "p_quadratic": p_q}
    print("  -> Quadratic adequate: using age_ord + age_sq")
    cols = ["age_ord", "age_sq"]
    return df[cols], cols, {"p_linear": p_lin, "p_quadratic": p_q}


def redundancy_check(df) -> pd.DataFrame:
    """Uncorrected Cramér's V for suspect pairs (flag > 0.6)."""
    rows = []
    for a, b in REDUNDANCY_PAIRS:
        v, p = cramers_v(df[a], df[b], bias_correction=False)
        rows.append({"var_a": a, "var_b": b, "cramers_v": v, "p": p,
                     "flag_gt_0.6": v > 0.6})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
def build_design_matrices(df: pd.DataFrame, verbose: bool = True) -> DesignSpec:
    y = get_target(df)
    check_rank_block(df)
    fe = engineer_features(df)

    if verbose:
        print("Age group counts:\n", fe["age_group"].value_counts().sort_index())
        print("\nWillingness rate by age group:\n",
              fe.groupby("age_group")[config.TARGET].mean().round(3))
        print("\nActivity group counts:\n", fe["activity_grp"].value_counts())
        print("\nMode group counts:\n", fe["mode_grp"].value_counts())

    phone_feats = phone_features(fe)
    exp_feat = expenditure_feature(fe)
    rank_feats = [f for f in config.RANK_BLOCK if f != config.DROP_REF]

    age_df, age_cols, age_diag = age_encoding(fe, y)

    act = pd.get_dummies(fe["activity_grp"], prefix="act").astype(int)
    act = act.drop(columns=["act_formal"])
    mode = pd.get_dummies(fe["mode_grp"], prefix="mode").astype(int)
    mode = mode.drop(columns=["mode_shared_public"])

    common = (["income", "education_level", "gender", "travel_assault",
               "mobility_issues", "hse_vehicle_ownership"]
              + exp_feat + phone_feats + rank_feats)

    X_A = pd.concat([fe[common + ["driving_license", "car_available"]],
                     age_df, act], axis=1).astype(float)
    X_B = pd.concat([fe[common + ["car_available"]], age_df, act, mode],
                    axis=1).astype(float)

    for_hire_n = int(fe["transport_to_work"].isin(FOR_HIRE_CODES).sum())
    diag = {"age": age_diag, "for_hire_n": for_hire_n,
            "redundancy": redundancy_check(fe),
            "income_x_education": pd.crosstab(fe["income"],
                                              fe["education_level"])}
    return DesignSpec(X_A, X_B, y, exp_feat, phone_feats, rank_feats,
                      age_cols, list(act.columns), list(mode.columns), diag)


def run_feature_engineering(df: pd.DataFrame, output_dir) -> DesignSpec:
    out = ensure_dir(output_dir / "feature_engineering")
    section("STEP 2 — Feature combination (Cramér's V + VIF)")
    spec = build_design_matrices(df)

    print(f"\n  Taxi + Uber respondents: {spec.diagnostics['for_hire_n']} "
          f"({'split out as for_hire' if spec.diagnostics['for_hire_n'] >= FOR_HIRE_MIN_N else 'merged into shared_public'})")

    print("\n  Cramér's V redundancy check (flag pairs > 0.6):")
    print(spec.diagnostics["redundancy"].round(3).to_string(index=False))
    save_table(spec.diagnostics["redundancy"], out / "cramers_v_pairs.csv",
               index=False)

    n_events = int(min(spec.y.sum(), (1 - spec.y).sum()))
    for name, X in [("A (explanatory)", spec.X_A), ("B (predictive)", spec.X_B)]:
        print(f"  Model {name}: {X.shape[1]} features, "
              f"EPP = {n_events / X.shape[1]:.1f} (minority n = {n_events})")

    for tag, X in [("A", spec.X_A), ("B", spec.X_B)]:
        vif = compute_vif(X)
        save_table(vif.to_frame("VIF"), out / f"vif_model_{tag}.csv")
        if tag == "A":
            print("\n  VIF (Model A) — investigate anything > 5:")
            print(vif.head(8).round(3).to_string())

    save_table(spec.diagnostics["income_x_education"],
               out / "income_x_education_crosstab.csv")
    save_table(spec.X_A, out / "design_matrix_A.csv", index=False)
    save_table(spec.X_B, out / "design_matrix_B.csv", index=False)
    return spec
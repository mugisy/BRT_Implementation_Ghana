"""STEP 1 — Correlation analyses.

Mixed measurement levels, method chosen per PAIR:
  * pure designations (income, age_group, education_level) -> Spearman
  * weights/counts & binary                                  -> Pearson
  * nominal (current_activity, transport_to_work, phone)     -> Cramér's V
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from . import config
from .io_utils import ensure_dir, save_figure, save_table, section
from .stats_utils import cramers_v


def classify_columns(df: pd.DataFrame) -> dict:
    keep = lambda cols: [c for c in cols if c in df.columns]  # noqa: E731
    return {
        "designation": keep(config.DESIGNATION_COLS),
        "weight_count": keep(config.WEIGHT_COUNT_COLS),
        "binary": keep(config.BINARY_COLS),
        "nominal": keep(config.NOMINAL_COLS),
    }


def mixed_correlation_matrix(df: pd.DataFrame, groups: dict):
    """Pearson by default, Spearman for any pair touching a designation."""
    designation = set(groups["designation"])
    cols = groups["designation"] + groups["weight_count"] + groups["binary"]
    n = len(cols)
    corr = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    pval = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    method = pd.DataFrame("-", index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            if c1 in designation or c2 in designation:
                r, p = spearmanr(df[c1], df[c2])
                m = "spearman"
            else:
                r, p = pearsonr(df[c1], df[c2])
                m = "pearson"
            corr.loc[c1, c2] = corr.loc[c2, c1] = r
            pval.loc[c1, c2] = pval.loc[c2, c1] = p
            method.loc[c1, c2] = method.loc[c2, c1] = m
    return corr, pval, method


def plot_mixed_correlation(corr, pval, path):
    labels = corr.apply(lambda col: col.map(lambda x: f"{x:.2f}"))
    for i in corr.index:
        for j in corr.columns:
            if i != j and pval.loc[i, j] < 0.05:
                labels.loc[i, j] += "*"
    fig = plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=labels, fmt="", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.4,
                cbar_kws={"shrink": 0.8, "label": "Correlation coefficient"},
                annot_kws={"size": 12})
    plt.title("Mixed-Method Correlation Matrix\n"
              "Pearson (weights/counts/binary) + Spearman (pure designations)"
              "  (* = p < 0.05)", fontsize=12, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_figure(fig, path)


def cramers_v_matrix(df: pd.DataFrame, cols):
    v = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                v.loc[c1, c2], p.loc[c1, c2] = 1.0, 0.0
            else:
                v.loc[c1, c2], p.loc[c1, c2] = cramers_v(df[c1], df[c2])
    return v, p


def plot_cramers_v(mat, path):
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, square=True, linewidths=0.4,
                cbar_kws={"shrink": 0.8, "label": "Cramér's V"},
                annot_kws={"size": 12})
    plt.title("Cramér's V — Nominal Variable Associations\n"
              "(Current_activity, transport_mode_to_work vs. all categorical "
              "features)", fontsize=12, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save_figure(fig, path)


def target_associations(df, corr, pval, method, nominal_cols,
                        target=config.TARGET) -> pd.DataFrame:
    rows = []
    for col in corr.index:
        if col == target:
            continue
        p = pval.loc[col, target]
        rows.append({"feature": col, "measure": method.loc[col, target],
                     "association": corr.loc[col, target], "p_value": p,
                     "significant_0.05": p < 0.05})
    for col in nominal_cols:
        v, p = cramers_v(df[col], df[target])
        rows.append({"feature": col, "measure": "cramers_v",
                     "association": v, "p_value": p,
                     "significant_0.05": p < 0.05})
    out = pd.DataFrame(rows)
    order = out["association"].abs().sort_values(ascending=False).index
    return out.loc[order].reset_index(drop=True)


def run_correlation_analysis(df: pd.DataFrame, output_dir) -> dict:
    """Run all three correlation analyses and save tables/figures."""
    out = ensure_dir(output_dir / "correlation")
    groups = classify_columns(df)

    section("STEP 1a — Mixed Pearson/Spearman correlation matrix")
    corr, pval, method = mixed_correlation_matrix(df, groups)
    save_table(corr, out / "main_correlation_matrix.csv")
    save_table(pval, out / "main_correlation_pvalues.csv")
    save_table(method, out / "main_correlation_methods.csv")
    plot_mixed_correlation(corr, pval, out / "main_correlation_matrix.png")
    print(f"  {len(corr)} variables; saved to {out}")

    section("STEP 1b — Cramér's V matrix (bias-corrected)")
    cat_cols = groups["nominal"] + groups["designation"] + groups["binary"]
    v, vp = cramers_v_matrix(df, cat_cols)
    save_table(v, out / "cramers_v_matrix.csv")
    save_table(vp, out / "cramers_v_pvalue_matrix.csv")
    plot_cramers_v(v, out / "cramers_v_matrix.png")

    section("STEP 1c — Associations with BRT willingness")
    tgt = target_associations(df, corr, pval, method, groups["nominal"])
    save_table(tgt, out / "target_correlations.csv", index=False)
    print(tgt.head(15).to_string(index=False))

    return {"corr": corr, "pvalues": pval, "methods": method,
            "cramers_v": v, "cramers_v_p": vp, "target": tgt}
"""STEP 5 — SHAP explainability for Gradient Boosting and Random Forest.

Explains TEST-SET predictions only (70/30 stratified split, RNG = 42).
Outputs: beeswarm, mean |SHAP| block comparison, dependence plots,
top interaction pairs (upper triangle, no self-pairs), interaction
summary plot and interaction-strength heatmaps.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from . import config
from .data import build_feature_sets, design_matrices, get_target
from .io_utils import ensure_dir, save_figure, save_table, section

DEPENDENCE_FEATURES = ["age_group", "hsehold_size"]


def positive_class(explanation):
    """Select the positive class for classifiers returning (n, p, 2)."""
    if explanation.values.ndim == 3:
        return explanation[:, :, 1]
    return explanation


def interaction_values(model, X) -> np.ndarray:
    inter = shap.TreeExplainer(model).shap_interaction_values(X)
    if isinstance(inter, list):
        inter = inter[1]
    elif inter.ndim == 4:
        inter = inter[:, :, :, 1]
    if inter.ndim != 3:
        raise ValueError(f"Unexpected interaction shape: {inter.shape}")
    return inter


def interaction_strength(inter, columns, zero_diagonal=True) -> pd.DataFrame:
    mat = np.abs(inter).mean(axis=0)
    if zero_diagonal:
        mat = mat.copy()
        np.fill_diagonal(mat, 0)
    return pd.DataFrame(mat, index=columns, columns=columns)


def top_interaction_pairs(strength: pd.DataFrame, n=10) -> pd.Series:
    mask = np.triu(np.ones(strength.shape), k=1).astype(bool)
    return strength.where(mask).stack().sort_values(ascending=False).head(n)


def mean_abs_shap_table(shap_values, columns, significant) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "group": ["sig" if f in significant else "NON-SIG" for f in columns],
    }).sort_values("mean_abs_shap", ascending=False)


def plot_interaction_heatmap(strength, title, path):
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(strength, annot=True, fmt=".3f", cmap="YlOrRd", square=True,
                linewidths=0.4, annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8,
                          "label": "Mean |SHAP interaction value|"})
    plt.title(title, fontsize=12, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    save_figure(fig, path)


def explain_model(model, name, X_te, significant, out_dir,
                  dependence_features=DEPENDENCE_FEATURES):
    out = ensure_dir(out_dir)
    section(f"SHAP — {name}")
    sv = positive_class(shap.TreeExplainer(model)(X_te))

    # 1) global importance + direction
    shap.plots.beeswarm(sv, max_display=X_te.shape[1], show=False)
    plt.title(f"SHAP summary — {name}, test set")
    plt.tight_layout()
    save_figure(plt.gcf(), out / "shap_beeswarm.png")

    # 2) significant vs excluded block
    tab = mean_abs_shap_table(sv, X_te.columns, significant)
    share = (tab.loc[tab.group == "NON-SIG", "mean_abs_shap"].sum()
             / tab["mean_abs_shap"].sum())
    print(tab.to_string(index=False))
    print(f"\n  Share of total |SHAP| from excluded block: {share:.1%}")
    save_table(tab, out / "mean_abs_shap.csv", index=False)

    # 3) dependence plots
    for feat in [f for f in dependence_features if f in X_te.columns]:
        shap.plots.scatter(sv[:, feat], color=sv, show=False)
        plt.tight_layout()
        save_figure(plt.gcf(), out / f"shap_dep_{feat}.png")

    # 4) interaction values
    inter = interaction_values(model, X_te)
    strength_off = interaction_strength(inter, X_te.columns, True)
    strength_full = interaction_strength(inter, X_te.columns, False)
    top = top_interaction_pairs(strength_off)
    print("\n  Top SHAP interaction pairs:\n", top)
    save_table(top.rename("mean_abs_interaction").to_frame(),
               out / "top_interaction_pairs.csv")

    shap.summary_plot(inter, X_te, show=False)
    plt.tight_layout()
    save_figure(plt.gcf(), out / "shap_interaction_summary.png")

    plot_interaction_heatmap(
        strength_full,
        f"SHAP Interaction Strength Matrix — {name}, test set\n"
        "(diagonal = main effect, off-diagonal = pairwise interaction)",
        out / "shap_interaction_matrix_full.png")
    plot_interaction_heatmap(
        strength_off,
        f"SHAP Pairwise Interaction Matrix — {name}, test set\n"
        "(diagonal zeroed — pure interactions only)",
        out / "shap_interaction_matrix_offdiag.png")
    return {"mean_abs": tab, "excluded_share": share, "top_pairs": top}


def run_shap_analysis(df, output_dir, feature_set="core", rng=config.RNG):
    """feature_set: 'core' (no activity/mode vars, as in the final RF/GB
    SHAP runs) or 'full' (all 10 significant features)."""
    significant = (config.SIGNIFICANT_CORE if feature_set == "core"
                   else config.SIGNIFICANT)
    y = get_target(df)
    fs = build_feature_sets(df, significant=significant)
    _, X_full = design_matrices(df, fs)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.3, stratify=y, random_state=rng)

    base = ensure_dir(output_dir / "shap" / feature_set)
    gb = GradientBoostingClassifier(random_state=rng).fit(X_tr, y_tr)
    rf = RandomForestClassifier(random_state=rng).fit(X_tr, y_tr)
    return {
        "GB": explain_model(gb, "GB", X_te, fs.significant, base / "GB"),
        "RF": explain_model(rf, "RF", X_te, fs.significant, base / "RF"),
    }
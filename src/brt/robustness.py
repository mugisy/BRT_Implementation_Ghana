"""STEP 6 — Robustness and improvements (pre-specified analysis plan).

Primary model set (Holm across 4 comparisons, endpoint = delta-AUC):
  Logistic regression, Logistic GAM, tuned RF, tuned GB.
Inner tuning: RandomizedSearchCV (capped to grid size -> small grids stay
exhaustive). Inference: Nadeau–Bengio corrected resampled t-test.
Robustness appendix (descriptive only): SVM-RBF, kNN, LightGBM, XGBoost.
Mechanism logic: GAM gain ~ tree gain -> nonlinearity;
                 GAM null, tree gain  -> interactions.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .consolidated_tests import (test1_lrt, test3_permutation_importance,
                                 test4_interaction_screen)
from .data import build_feature_sets, design_matrices, get_target
from .io_utils import ensure_dir, save_table, section
from .stats_utils import holm_correct, nadeau_bengio_ttest, nb_confint

try:
    from pygam import LogisticGAM, l as gam_l, s as gam_s
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _grid_size(grid):
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


def make_sklearn_arm(estimator, param_grid, scale, n_iter=config.N_ITER,
                     inner_splits=config.INNER_SPLITS, rng=config.RNG):
    """Return fit_predict(X_tr, y_tr, X_te) with inner random search."""
    def fit_predict(X_tr, y_tr, X_te):
        steps = ([("scale", StandardScaler())] if scale else []) \
            + [("clf", clone(estimator))]
        pipe = Pipeline(steps)
        if param_grid:
            grid = {f"clf__{k}": v for k, v in param_grid.items()}
            search = RandomizedSearchCV(
                pipe, grid, n_iter=min(n_iter, _grid_size(param_grid)),
                scoring="roc_auc", cv=inner_splits, n_jobs=-1,
                random_state=rng)
            est = search.fit(X_tr, y_tr).best_estimator_
        else:
            est = pipe.fit(X_tr, y_tr)
        if hasattr(est.named_steps["clf"], "predict_proba"):
            return est.predict_proba(X_te)[:, 1]
        return expit(est.decision_function(X_te))
    return fit_predict


def make_gam_arm():
    """Splines on features with >4 levels, linear terms otherwise; lam
    tuned by pygam's gridsearch (UBRE) on the training fold only."""
    def fit_predict(X_tr, y_tr, X_te):
        terms = None
        for i, col in enumerate(X_tr.columns):
            nun = X_tr[col].nunique()
            term = gam_s(i, n_splines=min(10, max(nun, 4))) if nun > 4 else gam_l(i)
            terms = term if terms is None else terms + term
        gam = LogisticGAM(terms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam.gridsearch(X_tr.values, y_tr.values,
                           lam=np.logspace(-3, 3, 7), progress=False)
        return gam.predict_proba(X_te.values)
    return fit_predict


def nested_cv_paired(arm, X_red, X_full, y, cv) -> pd.DataFrame:
    rows = []
    for tr, te in cv.split(X_full, y):
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        row = {}
        for tag, X in [("red", X_red), ("full", X_full)]:
            try:
                p = arm(X.iloc[tr], y_tr, X.iloc[te])
                row[f"auc_{tag}"] = roc_auc_score(y_te, p)
                row[f"ap_{tag}"] = average_precision_score(y_te, p)
                row[f"brier_{tag}"] = brier_score_loss(y_te, p)
            except Exception as exc:  # keep running, but say so
                warnings.warn(f"Fold failed ({tag}): {exc}")
                row[f"auc_{tag}"] = row[f"ap_{tag}"] = row[f"brier_{tag}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def report_arm(name, res, ratio, k, section_name="TEST 2"):
    section(f"{section_name} — {name} ({k} outer resamples, inner tuning)")
    summary = {}
    for metric, label in [("auc", "AUC  "), ("ap", "AP   "), ("brier", "Brier")]:
        d = res[f"{metric}_full"] - res[f"{metric}_red"]
        t, p, mean, _, _ = nadeau_bengio_ttest(d, ratio)
        lo, hi = nb_confint(d, ratio)
        print(f"  [{label}] Reduced {res[f'{metric}_red'].mean():.4f}  "
              f"Full {res[f'{metric}_full'].mean():.4f}  delta = {mean:+.4f} "
              f"(95% CI [{lo:+.4f}, {hi:+.4f}])  t_NB = {t:.3f}, p_NB = {p:.4f}")
        summary[metric] = {"delta": mean, "ci_lo": lo, "ci_hi": hi, "p": p}
    return summary


def primary_arms(n_iter, rng=config.RNG):
    arms = [
        ("Logistic Regression",
         make_sklearn_arm(LogisticRegression(max_iter=2000),
                          {"C": [0.01, 0.1, 1.0, 10.0]}, True, n_iter)),
        ("Random Forest (tuned)",
         make_sklearn_arm(RandomForestClassifier(n_estimators=500, random_state=rng),
                          {"max_depth": [3, 5, None],
                           "min_samples_leaf": [1, 5, 10]}, False, n_iter)),
        ("Gradient Boosting (tuned)",
         make_sklearn_arm(GradientBoostingClassifier(random_state=rng),
                          {"learning_rate": [0.01, 0.05, 0.1, 0.2],
                           "max_depth": [2, 3, 4, 5],
                           "n_estimators": [100, 300, 500],
                           "subsample": [0.7, 0.85, 1.0],
                           "min_samples_leaf": [1, 5, 10],
                           "max_features": ["sqrt", "log2", None]},
                          False, n_iter)),
    ]
    if HAS_PYGAM:
        arms.insert(1, ("Logistic GAM", make_gam_arm()))
    else:
        warnings.warn("pygam not installed -> GAM arm skipped.")
    return arms


def appendix_arms(n_iter, rng=config.RNG):
    arms = [
        ("SVM (RBF)", make_sklearn_arm(SVC(probability=True, random_state=rng),
                                       {"C": [0.1, 1, 10], "gamma": ["scale", 0.1]},
                                       True, n_iter)),
        ("kNN", make_sklearn_arm(KNeighborsClassifier(),
                                 {"n_neighbors": [5, 11, 21]}, True, n_iter)),
    ]
    if HAS_LGBM:
        arms.append(("LightGBM", make_sklearn_arm(
            LGBMClassifier(random_state=rng, verbose=-1),
            {"min_child_samples": [10, 20, 40], "num_leaves": [7, 15],
             "learning_rate": [0.05, 0.1]}, False, n_iter)))
    if HAS_XGB:
        arms.append(("XGBoost", make_sklearn_arm(
            XGBClassifier(random_state=rng, eval_metric="logloss"),
            {"max_depth": [2, 3], "learning_rate": [0.05, 0.1],
             "n_estimators": [100, 300]}, False, n_iter)))
    return arms


def mechanism_readout(names, p_holm, alpha=config.ALPHA):
    if "Logistic GAM" not in names:
        return "GAM arm unavailable — mechanism not decomposed."
    gam_sig = p_holm[names.index("Logistic GAM")] < alpha
    gb_sig = p_holm[names.index("Gradient Boosting (tuned)")] < alpha
    if gam_sig and gb_sig:
        return "GAM and GB both gain: signal includes NONLINEAR main effects."
    if gb_sig:
        return "GB gains but GAM does not: signal carried by INTERACTIONS."
    if gam_sig:
        return "GAM gains but GB does not: smooth nonlinear main effects."
    return "No robust predictive gain after tuning and corrected inference."


def run_robustness(df, output_dir, n_splits=config.N_SPLITS,
                   n_repeats=config.N_REPEATS, n_iter=config.N_ITER,
                   run_appendix=True, rng=config.RNG):
    out = ensure_dir(output_dir / "robustness")
    y = get_target(df)
    fs = build_feature_sets(df, significant=config.SIGNIFICANT)
    X_red, X_full = design_matrices(df, fs)
    print(f"n = {len(y)}, positives = {y.mean():.1%} -> AUC, AP, Brier.")

    t1 = test1_lrt(X_red, X_full, y, fs.non_significant, df, fs.significant)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=rng)
    k, ratio = n_splits * n_repeats, 1.0 / (n_splits - 1)

    section("TEST 2 — PRIMARY ANALYSIS (pre-specified model set)")
    primary = {}
    for name, arm in primary_arms(n_iter, rng):
        res = nested_cv_paired(arm, X_red, X_full, y, cv)
        save_table(res, out / f"folds_{name.replace(' ', '_')}.csv", index=False)
        primary[name] = report_arm(name, res, ratio, k)

    names = list(primary)
    p_auc = [primary[n]["auc"]["p"] for n in names]
    p_holm = holm_correct(p_auc)
    section("TEST 2 — Primary family, Holm-adjusted (delta-AUC)")
    rows = []
    for n, p_raw, p_adj in zip(names, p_auc, p_holm):
        a = primary[n]["auc"]
        print(f"  {n:<26} dAUC = {a['delta']:+.4f} [{a['ci_lo']:+.4f},"
              f"{a['ci_hi']:+.4f}]  p = {p_raw:.4f}  p_Holm = {p_adj:.4f}"
              f"{' **' if p_adj < config.ALPHA else ''}")
        rows.append({"model": n, **a, "p_holm": p_adj})
    family = pd.DataFrame(rows)
    save_table(family, out / "primary_family_holm.csv", index=False)
    readout = mechanism_readout(names, p_holm)
    print(f"\n  Mechanism read-out: {readout}")

    if run_appendix:
        section("TEST 2R — ROBUSTNESS APPENDIX (descriptive only)", "#")
        app = {}
        for name, arm in appendix_arms(n_iter, rng):
            res = nested_cv_paired(arm, X_red, X_full, y, cv)
            app[name] = report_arm(name, res, ratio, k, "TEST 2R [appendix]")
        save_table(pd.DataFrame({n: v["auc"] for n, v in app.items()}).T,
                   out / "appendix_auc.csv")

    X_tr, _, y_tr, _ = train_test_split(X_full, y, test_size=0.3,
                                        stratify=y, random_state=rng)
    rf_grid = GridSearchCV(
        RandomForestClassifier(n_estimators=500, random_state=rng),
        {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]},
        scoring="roc_auc", cv=config.INNER_SPLITS, n_jobs=-1).fit(X_tr, y_tr)
    section(f"TEST 3 — Permutation importance (tuned RF: {rf_grid.best_params_})")
    t3 = test3_permutation_importance(X_full, y, fs.significant,
                                      model=rf_grid.best_estimator_)
    save_table(t3, out / "test3_permutation_importance_tuned_rf.csv", index=False)

    section("TEST 4 — Interaction screening")
    t4 = test4_interaction_screen(X_full, y, fs.non_significant, fs.significant)
    save_table(t4, out / "test4_interactions.csv", index=False)
    return {"test1": t1, "primary": family, "mechanism": readout,
            "test3": t3, "test4": t4}
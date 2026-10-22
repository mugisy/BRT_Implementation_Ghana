"""Command-line entry point: run any subset of the pipeline steps."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from . import config  # noqa: E402
from .consolidated_tests import run_consolidated_tests  # noqa: E402
from .correlation import run_correlation_analysis  # noqa: E402
from .data import load_data  # noqa: E402
from .explanatory_models import run_explanatory_analysis  # noqa: E402
from .feature_engineering import run_feature_engineering  # noqa: E402
from .robustness import run_robustness  # noqa: E402
from .shap_analysis import run_shap_analysis  # noqa: E402

STEPS = ["correlation", "features", "explanatory", "consolidated",
         "shap", "robustness"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BRT willingness pipeline")
    p.add_argument("--data", type=Path, default=config.DATA_PATH)
    p.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--steps", nargs="+", choices=STEPS + ["all"],
                   default=["all"])
    p.add_argument("--shap-feature-set", choices=["core", "full"],
                   default="core")
    p.add_argument("--no-appendix", action="store_true",
                   help="skip the robustness appendix (SVM, kNN, LGBM, XGB)")
    p.add_argument("--quick", action="store_true",
                   help="fewer CV repeats / search draws for a fast run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    steps = STEPS if "all" in args.steps else args.steps
    df = load_data(args.data)
    out = args.output
    n_repeats = 2 if args.quick else config.N_REPEATS
    n_iter = 10 if args.quick else config.N_ITER

    spec = None
    if "correlation" in steps:
        run_correlation_analysis(df, out)
    if "features" in steps:
        spec = run_feature_engineering(df, out)
    if "explanatory" in steps:
        run_explanatory_analysis(df, out, spec=spec)
    if "consolidated" in steps:
        run_consolidated_tests(df, out, n_repeats=n_repeats)
    if "shap" in steps:
        run_shap_analysis(df, out, feature_set=args.shap_feature_set)
    if "robustness" in steps:
        run_robustness(df, out, n_repeats=n_repeats, n_iter=n_iter,
                       run_appendix=not args.no_appendix)
    print(f"\nDone. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
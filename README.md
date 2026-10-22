# BRT_Implementation_Ghana

A modeling study of how commuters in Kumasi, Ghana would respond to a Bus
Rapid Transit (BRT) system. Using survey data, the pipeline
(1) finds the most important BRT success factors and (2) estimates how
they affect willingness to switch to BRT.

## Pipeline

| Step | Module | Content |
|---|---|---|
| 1 | `correlation.py` | Mixed Pearson/Spearman matrix, bias-corrected Cramér's V matrix, associations with `brt_willingness` |
| 2 | `feature_engineering.py` | Recoding (activity, mode, phone, age), private-mobility composite, Cramér's V redundancy checks, age-encoding tests, VIF, design matrices A/B |
| 3 | `explanatory_models.py` | Odds ratios, block LR tests, mediation contrast A→B, CV AUC, Holm-corrected interaction screen |
| 4 | `consolidated_tests.py` | Tests 1–4: LRTs, repeated-CV gain, permutation importance, interaction screening |
| 5 | `shap_analysis.py` | SHAP for GB and RF: beeswarm, block share, dependence plots, interaction pairs/heatmaps |
| 6 | `robustness.py` | Pre-specified nested-CV plan (LR, GAM, tuned RF/GB), Nadeau–Bengio + Holm, mechanism read-out, robustness appendix |

## Installation
```bash
git clone https://github.com/mugisy/BRT_Implementation_Ghana.git
cd BRT_Implementation_Ghana
python -m venv .venv && source .venv/bin/activate
pip install -e ".[extras,dev]"
```

## Data
Put `brt_clean.csv` in `data/`. Variable coding after cleaning:
- income: <GH₵500=0 … >GH₵5000=4; gender: F=0, M=1; age_group: 12–18=0 … >65=6
- driving_license, travel_assault, mobility_issues, brt_willingness: No=0, Yes=1
- phone_ownership: No=0, Yes+Internet=1, Yes no Internet=2
- comfort, reliability, affordability, safety, travel_time_factor: ranks 1–5
  (compositional; `safety` is dropped as the reference)

## Usage
```bash
brt-run                                     # all steps
brt-run --steps correlation features explanatory
brt-run --steps shap --shap-feature-set full
brt-run --steps robustness --no-appendix
brt-run --quick                             # fast smoke run
pytest -q
```
All tables and figures are written to `outputs/<step>/`.

## Notes
- The SHAP step uses `core` features by default (no `current_activity` or
  `transport_to_work`); this reproduces the final RF/GB SHAP figures.
- Test 2 in step 4 reports both the naive paired t-test and the
  Nadeau–Bengio corrected test. Use the corrected test for inference.
- The GAM, LightGBM and XGBoost arms run only if the optional packages are
  installed.
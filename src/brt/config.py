"""Central configuration: paths, seeds, and feature groups."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "brt_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RNG = 42
ALPHA = 0.05
TARGET = "brt_willingness"

# ---------------------------------------------------------------------
# Feature groups used by the modeling tests
# ---------------------------------------------------------------------
SIGNIFICANT = [
    "income", "comfort", "hse_vehicle_ownership", "monthly_transport_exp",
    "gender", "driving_license", "travel_assault", "mobility_issues",
    "current_activity", "transport_to_work",
]
# Core set (no activity / mode variables) – used for the final RF/GB SHAP runs
SIGNIFICANT_CORE = [f for f in SIGNIFICANT
                    if f not in ("current_activity", "transport_to_work")]

NON_SIGNIFICANT = [
    "age_group", "education_level", "travel_time_factor", "reliability",
    "affordability", "safety", "hsehold_size",
]

# Compositional ranking block (ranks sum to a constant) – drop one reference
RANK_BLOCK = ["comfort", "travel_time_factor", "reliability",
              "affordability", "safety"]
DROP_REF = "safety"

# Post-hoc exploratory candidates (Test 1b)
CANDIDATES = ["age_group", "hsehold_size"]

# ---------------------------------------------------------------------
# Measurement levels for the correlation analysis
# ---------------------------------------------------------------------
DESIGNATION_COLS = ["income", "age_group", "education_level"]
WEIGHT_COUNT_COLS = [
    "travel_time_factor", "comfort", "reliability", "affordability", "safety",
    "hsehold_size", "hse_vehicle_ownership", "monthly_transport_exp",
]
BINARY_COLS = ["gender", "disability", "driving_license", "travel_assault",
               "mobility_issues", "brt_willingness"]
NOMINAL_COLS = ["current_activity", "transport_to_work", "phone_ownership"]

# ---------------------------------------------------------------------
# Cross-validation defaults
# ---------------------------------------------------------------------
N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 3
N_ITER = 60
"""Data loading and feature-set construction."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Put brt_clean.csv in data/ "
            "or pass --data.")
    return pd.read_csv(path)


def get_target(df: pd.DataFrame, target: str = config.TARGET) -> pd.Series:
    return df[target].astype(int)


def check_rank_block(df: pd.DataFrame, rank_block=config.RANK_BLOCK) -> int:
    """Verify the ranking block is compositional (constant row sum)."""
    sums = df[rank_block].sum(axis=1)
    if sums.nunique() != 1:
        raise ValueError("Rank block is not compositional — do not drop a "
                         "reference variable.")
    return int(sums.iloc[0])


@dataclass
class FeatureSets:
    significant: List[str]
    non_significant: List[str]

    @property
    def all(self) -> List[str]:
        return self.significant + self.non_significant


def build_feature_sets(df: pd.DataFrame,
                       significant=config.SIGNIFICANT,
                       non_significant=config.NON_SIGNIFICANT,
                       drop_ref: str = config.DROP_REF) -> FeatureSets:
    """Drop the compositional reference and keep columns present in df."""
    check_rank_block(df)
    sig = [f for f in significant if f != drop_ref and f in df.columns]
    non = [f for f in non_significant if f != drop_ref and f in df.columns]
    return FeatureSets(sig, non)


def design_matrices(df: pd.DataFrame, fs: FeatureSets):
    """Return (X_reduced, X_full) as float DataFrames."""
    return (df[fs.significant].astype(float).copy(),
            df[fs.all].astype(float).copy())
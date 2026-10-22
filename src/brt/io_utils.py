"""Small helpers for console reporting and saving outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def section(title: str, char: str = "=", width: int = 60) -> None:
    print("\n" + char * width)
    print(title)


def save_table(df: pd.DataFrame, path, index: bool = True) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def save_figure(fig, path, dpi: int = 600) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
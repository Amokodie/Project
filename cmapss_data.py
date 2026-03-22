"""Load NASA C-MAPSS FD001–FD004 train/test and RUL files (26 columns, no header)."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# Standard C-MAPSS layout: unit, cycle, 3 settings, 21 sensors
COL_NAMES = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Default location you provided (Windows). Override with env CMAPSS_DATA_DIR.
DEFAULT_EXTERNAL_CMAPSS = os.path.join(
    r"E:\6.+Turbofan+Engine+Degradation+Simulation+Data+Set",
    "6. Turbofan Engine Degradation Simulation Data Set",
    "CMAPSSData",
)


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_local_test_path() -> str:
    return os.path.join(_project_root(), "data", "test_FD001.txt")


def resolve_cmapss_root(explicit: Optional[str] = None) -> str:
    """Prefer explicit path, then CMAPSS_DATA_DIR, then E: default, then local data folder."""
    if explicit and os.path.isdir(explicit):
        return os.path.normpath(explicit)
    env = os.environ.get("CMAPSS_DATA_DIR", "").strip()
    if env and os.path.isdir(env):
        return os.path.normpath(env)
    if os.path.isdir(DEFAULT_EXTERNAL_CMAPSS):
        return os.path.normpath(DEFAULT_EXTERNAL_CMAPSS)
    return os.path.join(_project_root(), "data")


def cmapss_file(root: str, split: str, fd: str) -> str:
    """split: 'train' | 'test'; fd: 'FD001' … 'FD004'."""
    split = split.lower().strip()
    fd = fd.upper().strip()
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    return os.path.join(root, f"{split}_{fd}.txt")


def rul_file(root: str, fd: str) -> str:
    return os.path.join(root, f"RUL_{fd.upper()}.txt")


def readme_path(root: str) -> str:
    p = os.path.join(root, "readme.txt")
    return p if os.path.isfile(p) else ""


ALL_FD_IDS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")


def list_available_datasets(root: str) -> list[str]:
    """FD ids that have at least one of train_FD00x.txt or test_FD00x.txt under root."""
    if not root or not os.path.isdir(root):
        return []
    found: list[str] = []
    for fd in ALL_FD_IDS:
        if os.path.isfile(cmapss_file(root, "train", fd)) or os.path.isfile(
            cmapss_file(root, "test", fd)
        ):
            found.append(fd)
    return found


def load_cmapss_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 26:
        raise ValueError(f"Expected 26 columns, got {df.shape[1]} in {path}")
    df.columns = list(COL_NAMES)
    return df


def load_rul(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    s = pd.read_csv(path, header=None).iloc[:, 0].astype(float).values
    return s


def generate_synthetic_fd001(
    n_units: int = 24,
    seed: int = 42,
    max_cycles: int = 200,
) -> pd.DataFrame:
    """Synthetic turbofan-like trajectories when no NASA files are available."""
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for u in range(1, n_units + 1):
        n = int(rng.integers(80, max_cycles + 1))
        t = np.arange(1, n + 1)
        progress = t / n
        s1 = rng.normal(0.0, 0.02, size=n) + 0.0018 * t
        s2 = rng.normal(0.0, 0.02, size=n) + 0.0005 * t
        s3 = rng.normal(0.0, 0.01, size=n)
        base = rng.normal(0, 0.5, size=21)
        drift = np.outer(progress, rng.uniform(0.5, 2.5, size=21))
        noise = rng.normal(0, 0.03, size=(n, 21))
        sensors = base + drift + noise
        sensors[:, 5] += 0.8 * progress
        sensors[:, 10] -= 0.4 * progress
        unit_col = np.full(n, u)
        block = np.column_stack([unit_col, t, s1, s2, s3, sensors])
        rows.append(block)
    data = np.vstack(rows)
    return pd.DataFrame(data, columns=list(COL_NAMES))


def load_test_fd001(path: Optional[str] = None) -> pd.DataFrame:
    """Backward-compatible loader for app default."""
    path = path or default_local_test_path()
    if os.path.isfile(path):
        return load_cmapss_table(path)
    return generate_synthetic_fd001()


def sensor_labels_for_ui() -> dict[str, str]:
    return {
        "sensor_1": "T2 (fan inlet temp.)",
        "sensor_2": "T24 (LPC outlet temp.)",
        "sensor_3": "T30 (HPC outlet temp.)",
        "sensor_4": "T50 (LPT outlet temp.)",
        "sensor_5": "P2 (fan inlet press.)",
        "sensor_6": "P15 (bypass duct P)",
        "sensor_7": "P30 (HPC outlet press.)",
        "sensor_8": "Nf (physical fan speed)",
        "sensor_9": "Nc (physical core speed)",
        "sensor_10": "epr (engine pressure ratio)",
        "sensor_11": "Ps30 (static HPC outlet P)",
        "sensor_12": "phi (ratio fuel flow / Ps30)",
        "sensor_13": "NRf (corrected fan speed)",
        "sensor_14": "NRc (corrected core speed)",
        "sensor_15": "BPR (bypass ratio)",
        "sensor_16": "farB (burner fuel-air ratio)",
        "sensor_17": "htBleed (bleed enthalpy)",
        "sensor_18": "NF fan speed (demanded)",
        "sensor_19": "NC core speed (demanded)",
        "sensor_20": "Nf (corrected) / NRf",
        "sensor_21": "Nc (corrected) / NRc",
    }


def fd_label(fd: str) -> str:
    return fd.upper().strip()

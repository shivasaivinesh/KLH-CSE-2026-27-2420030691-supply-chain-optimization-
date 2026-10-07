"""
Reproducible dataset acquisition.

The three benchmark datasets used by this project are large (≈210 MB in total)
and are therefore not versioned inside this repository.  This script fetches
them from their public mirrors so that the evaluation is reproducible.

Run:
    python src/fetch_data.py

The files land in $SC_DATA_ROOT/raw (default: ./data/raw).

Datasets
--------
D1  DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
    https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
D2  Store Item Demand Forecasting (Kaggle)
    https://www.kaggle.com/competitions/demand-forecasting-kernels-only
D3  UCI Online Retail II (UCI ML Repository, ID 502)
    https://archive.ics.uci.edu/dataset/502/online+retail+ii

If the automatic mirrors are unavailable, download the files manually and place
them in data/raw with the names listed in src/config.py.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MIRRORS = [
    {
        "name": "D1 DataCo Smart Supply Chain",
        "repo": "https://github.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS.git",
        "wanted": {
            "DataCoSupplyChainDataset.csv": config.DATACO_CSV,
        },
    },
    {
        "name": "D2 Store Item Demand Forecasting",
        "repo": "https://github.com/jgonzalezab/Store-Item-Demand-Forecasting.git",
        "wanted": {
            "Data/train.csv": config.STORE_DEMAND_CSV,
        },
    },
    {
        "name": "D3 UCI Online Retail II",
        "repo": "https://github.com/newtonepv/time_series_analysis.git",
        "wanted": {
            "online_retail_2/online_retail_II.csv": config.ONLINE_RETAIL_CSV,
        },
    },
]


def _clone(repo: str, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", repo, str(dest)],
        check=True,
    )


def fetch() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for spec in MIRRORS:
        missing = [s for s, d in spec["wanted"].items() if not Path(d).exists()]
        if not missing:
            print(f"[skip] {spec['name']} - already present")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "repo"
            print(f"[fetch] {spec['name']} ...")
            try:
                _clone(spec["repo"], tmp_path)
            except subprocess.CalledProcessError:
                print(f"  !! could not clone {spec['repo']} - download manually")
                continue
            for src_rel, dest in spec["wanted"].items():
                src = tmp_path / src_rel
                if src.exists():
                    shutil.copy2(src, dest)
                    print(f"  -> {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
                else:
                    print(f"  !! {src_rel} not found in mirror")
    print("\nDone. Raw data directory:", config.RAW_DIR)


if __name__ == "__main__":
    fetch()

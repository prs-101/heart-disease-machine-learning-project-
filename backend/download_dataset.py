"""
Download the UCI Heart Disease dataset (Cleveland subset) from the official
UCI Machine Learning Repository and save it as data/raw/heart_disease.csv.

Usage (from the project root):
    python download_dataset.py                 # download (skips if already present)
    python download_dataset.py --force         # re-download
    python download_dataset.py --from-file C:\\Users\\you\\Downloads\\processed.cleveland.data

The --from-file option is the MANUAL alternative if the automatic download
fails (e.g. no internet, firewall). This script never generates synthetic data.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from preprocessing.preprocessing_utils import RAW_COLUMNS, RAW_DATA_PATH, RAW_DIR, ensure_dirs  # noqa: E402

# Official UCI sources (tried in order)
UCI_DIRECT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
ZIP_MEMBER = "processed.cleveland.data"

MANUAL_INSTRUCTIONS = f"""
------------------------------------------------------------------
AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD STEPS
------------------------------------------------------------------
1. Open {UCI_PAGE} in your browser.
2. Click "Download" (you get heart+disease.zip).
3. Extract the zip and find the file:  processed.cleveland.data
4. Run (adjust the path to where you extracted it):
     python download_dataset.py --from-file "C:\\path\\to\\processed.cleveland.data"
   Alternatively open this link directly and save the page as
   processed.cleveland.data:
     {UCI_DIRECT_URL}
------------------------------------------------------------------
"""


def _fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "heart-disease-ml/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_cleveland(raw_bytes: bytes) -> pd.DataFrame:
    """Parse the comma-separated, header-less UCI file. '?' = missing."""
    df = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=RAW_COLUMNS,
                     na_values=["?"], skipinitialspace=True)
    df = df.dropna(how="all")
    return df


def validate(df: pd.DataFrame) -> None:
    """Sanity checks so a wrong file is never silently used."""
    if df.shape[1] != 14:
        raise ValueError(f"Expected 14 columns, found {df.shape[1]}.")
    if not 250 <= len(df) <= 350:
        raise ValueError(f"Expected about 303 rows (Cleveland subset), found {len(df)}.")
    bad_target = set(df["num"].dropna().unique()) - {0, 1, 2, 3, 4}
    if bad_target:
        raise ValueError(f"Unexpected target values in 'num': {bad_target}")


def save_raw(df: pd.DataFrame) -> None:
    ensure_dirs(RAW_DIR)
    # Missing values are written back as '?' so the raw file stays faithful
    # to the original UCI format. Cleaning happens in preprocessing/clean_data.py.
    df.to_csv(RAW_DATA_PATH, index=False, na_rep="?")
    n_missing = int(df.isna().sum().sum())
    print(f"Saved {len(df)} rows x {df.shape[1]} columns -> {RAW_DATA_PATH}")
    print(f"Missing values ('?') in raw data: {n_missing}")


def download_from_uci() -> pd.DataFrame:
    errors = []
    print(f"Downloading from {UCI_DIRECT_URL}")
    try:
        return parse_cleveland(_fetch(UCI_DIRECT_URL))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        errors.append(f"direct file: {exc}")
        print(f"  direct download failed ({exc}); trying the official zip archive ...")

    try:
        archive = zipfile.ZipFile(io.BytesIO(_fetch(UCI_ZIP_URL, timeout=60)))
        member = next(n for n in archive.namelist() if n.endswith(ZIP_MEMBER))
        return parse_cleveland(archive.read(member))
    except (urllib.error.URLError, OSError, ValueError, StopIteration, zipfile.BadZipFile) as exc:
        errors.append(f"zip archive: {exc}")

    raise RuntimeError("All official UCI sources failed:\n  " + "\n  ".join(errors))


def main(force: bool = False, from_file: str | None = None) -> Path:
    if from_file:
        src = Path(from_file).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"File not found: {src}")
        print(f"Loading manually downloaded file: {src}")
        df = parse_cleveland(src.read_bytes())
        validate(df)
        save_raw(df)
        return RAW_DATA_PATH

    if RAW_DATA_PATH.exists() and not force:
        print(f"Dataset already present: {RAW_DATA_PATH} (use --force to re-download)")
        return RAW_DATA_PATH

    try:
        df = download_from_uci()
        validate(df)
    except Exception as exc:  # noqa: BLE001 - show a friendly message for any failure
        print(f"\nERROR: {exc}")
        print(MANUAL_INSTRUCTIONS)
        raise SystemExit(1)

    save_raw(df)
    return RAW_DATA_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--from-file", help="path to a manually downloaded processed.cleveland.data")
    args = parser.parse_args()
    main(force=args.force, from_file=args.from_file)

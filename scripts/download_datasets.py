# -*- coding: utf-8 -*-
"""
download_datasets.py
Downloads UCI Heart Disease site files (Cleveland, Hungary, Switzerland, VA Long Beach),
the Statlog Heart dataset, and attempts to retrieve the Framingham educational CSV.
Writes SHA-256 checksums and provenance notes to data/raw/CHECKSUMS.md.
All raw files are stored unmodified.
"""
import hashlib
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
UCI_DIR = RAW_DIR / "uci"
STATLOG_DIR = RAW_DIR / "statlog"
FRAMINGHAM_DIR = RAW_DIR / "framingham"

# Official UCI ML repository mirrors
UCI_FILES = {
    "cleveland.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungary.data":   "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va.data":        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}
STATLOG_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"

# Framingham: educational CSV circulated through Kaggle and GitHub tutorials.
# We attempt the GauravPadawe GitHub mirror; provenance is UNVERIFIED.
FRAMINGHAM_URL = "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_file(url: str, dest: Path) -> str:
    """Download url to dest. Returns SHA-256 hex string."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    sha = sha256_of_bytes(resp.content)
    print(f"  OK  {dest.name}  (SHA256: {sha[:16]}...)")
    return sha


def main():
    results = []

    # --- UCI Heart Disease files ---
    print("Downloading UCI Heart Disease datasets...")
    for fname, url in UCI_FILES.items():
        dest = UCI_DIR / fname
        if dest.exists():
            sha = sha256_of_bytes(dest.read_bytes())
            print(f"  EXISTS {fname}  (SHA256: {sha[:16]}...)")
        else:
            sha = download_file(url, dest)
        results.append(("uci/" + fname, url, sha, "UCI ML Repository (public domain)"))

    # --- Statlog ---
    print("Downloading Statlog Heart dataset...")
    statlog_dest = STATLOG_DIR / "heart.dat"
    if statlog_dest.exists():
        sha = sha256_of_bytes(statlog_dest.read_bytes())
        print(f"  EXISTS heart.dat  (SHA256: {sha[:16]}...)")
    else:
        sha = download_file(STATLOG_URL, statlog_dest)
    results.append(("statlog/heart.dat", STATLOG_URL, sha, "UCI ML Repository (public domain)"))

    # --- Framingham ---
    print("Attempting Framingham educational CSV download...")
    fram_dest = FRAMINGHAM_DIR / "framingham.csv"
    if fram_dest.exists():
        sha = sha256_of_bytes(fram_dest.read_bytes())
        print(f"  EXISTS framingham.csv  (SHA256: {sha[:16]}...)")
        results.append(("framingham/framingham.csv", FRAMINGHAM_URL, sha,
                         "UNVERIFIED - GitHub educational mirror; original source and license unknown"))
    else:
        try:
            sha = download_file(FRAMINGHAM_URL, fram_dest)
            results.append(("framingham/framingham.csv", FRAMINGHAM_URL, sha,
                             "UNVERIFIED - GitHub educational mirror; original source and license unknown"))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append(("framingham/framingham.csv", FRAMINGHAM_URL, "DOWNLOAD_FAILED",
                             "UNRESOLVED - Could not retrieve; provenance and license unknown"))

    # --- Write CHECKSUMS.md ---
    checksum_path = RAW_DIR / "CHECKSUMS.md"
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write("# Raw Data Checksums and Provenance\n\n")
        f.write("| File | Source URL | SHA-256 | License / Notes |\n")
        f.write("|------|------------|---------|----------------|\n")
        for fpath, url, sha, note in results:
            f.write(f"| {fpath} | {url} | `{sha}` | {note} |\n")
        f.write("\n")
        f.write("## Framingham Provenance Note\n\n")
        f.write("The 4,240-row `framingham.csv` is an educational subset derived from the "
                "Framingham Heart Study longitudinal cohort. It circulates on Kaggle "
                "(e.g., dataset by Ashish Bhardwaj) and in GitHub tutorial repositories. "
                "The exact original source, redistribution licence, and relationship to "
                "the NHLBI BIOLINCC official teaching dataset are **UNVERIFIED**. "
                "This file is strictly isolated in Pipeline B and must not be merged "
                "with Pipeline A (UCI diagnostic datasets). Do not use for model training "
                "until provenance is confirmed.\n")

    print(f"\nChecksums written to: {checksum_path}")


if __name__ == "__main__":
    main()

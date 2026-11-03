# Dataset Provenance Report

> HeartGuard AI — Week 2 | Verified programmatically

## 1. UCI Heart Disease Datasets (Pipeline A)

### Common Metadata
- **Repository:** UCI Machine Learning Repository — Heart Disease Dataset (ID: 45)
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Principal Investigators:** Andras Janosi (Hungary), William Steinbrunn (Switzerland), Matthias Pfisterer (Switzerland), Robert Detrano (VA Long Beach / Cleveland)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Usage rights:** Freely usable for research and educational purposes with attribution.
- **Raw file base URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

### Cleveland (processed.cleveland.data)
| Field | Value |
|-------|-------|
| Rows | 303 |
| Columns | 14 (13 features + target `num`) |
| Missing values | Yes (`?` coded) — primarily `ca` and `thal` |
| Target (`num`) | 0 = no disease; 1-4 = disease severity |
| SHA-256 | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| Notes | Derivation cohort for Pipeline A model training and internal validation |

### Hungary (processed.hungarian.data)
| Field | Value |
|-------|-------|
| Rows | 294 |
| Columns | 14 (13 features + target `num`) |
| Missing values | High — `slope`, `ca`, `thal`, `chol`, `fbs` frequently missing |
| Target (`num`) | 0 = no disease; 1-4 = disease severity |
| SHA-256 | `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796` |
| Notes | External validation cohort (zero-shot only; must not influence training) |

### Switzerland (processed.switzerland.data)
| Field | Value |
|-------|-------|
| Rows | 123 |
| Columns | 14 (13 features + target `num`) |
| Missing values | High — `chol`, `fbs`, `restecg`, `slope`, `ca`, `thal` frequently missing |
| Zero-coded `chol` | Present; cause of zero coding is NOT verified from dataset documentation |
| Target (`num`) | 0 = no disease; 1-4 = disease severity |
| SHA-256 | `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314` |
| Notes | External validation cohort (zero-shot only; must not influence training) |

### VA Long Beach (processed.va.data)
| Field | Value |
|-------|-------|
| Rows | 200 |
| Columns | 14 (13 features + target `num`) |
| Missing values | Moderate — `slope`, `ca`, `thal`, `chol` affected |
| Target (`num`) | 0 = no disease; 1-4 = disease severity |
| SHA-256 | `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f` |
| Notes | External validation cohort (zero-shot only; must not influence training) |

---

## 2. Statlog Heart Dataset

| Field | Value |
|-------|-------|
| Repository | UCI ML Repository — Statlog (Heart) Dataset (ID: 145) |
| URL | https://archive.ics.uci.edu/dataset/145/statlog+heart |
| Source file | heart.dat (space-separated) |
| Rows | 270 |
| Columns | 14 (13 features + binary target) |
| Missing values | None documented |
| Target | 1 = absence of disease; 2 = presence of disease |
| SHA-256 | `f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9` |
| License | CC BY 4.0 |

### Programmatic Comparison with Cleveland (Week 2 Finding)

Verification script: `scripts/verify_statlog_duplicates.py`

| Metric | Result |
|--------|--------|
| Exact string-key matches (Cleveland -> Statlog) | **270 of 303 Cleveland rows** |
| Exact string-key matches (Statlog -> Cleveland) | **270 of 270 Statlog rows (100%)** |
| Near matches (Euclidean distance > 0 and <= 1.0) | 0 |
| Unmatched Cleveland rows | 33 |
| Unmatched Statlog rows | 0 |

**Interpretation:** Every Statlog record has an exact string-key counterpart in Cleveland across all 13 common features. The 33 unmatched Cleveland rows contain missing values (`?`) which prevent exact key matching. This provides strong programmatic evidence that Statlog is a subset of Cleveland (270 of 303 rows), but feature coding differences and the binary vs. ordinal target definition require clinical review before this is treated as established fact.

**Status: PENDING clinical coding review** — do not treat Statlog as confirmed Cleveland subset until reviewed.

---

## 3. Framingham Educational Dataset (Pipeline B)

> **ISOLATION REQUIREMENT:** This dataset is strictly isolated in Pipeline B (prognostic).
> It must NOT be merged with Pipeline A (UCI diagnostic datasets) under any circumstances.

| Field | Value |
|-------|-------|
| File | `data/raw/framingham/framingham.csv` |
| Rows | 4,238 (after download; nominal 4,240) |
| Columns | 16 (15 predictors + `TenYearCHD` target) |
| Target | `TenYearCHD` — 10-year coronary heart disease risk (binary: 0/1) |
| SHA-256 | `2c0e57dc0361b420...` (see CHECKSUMS.md for full hash) |
| Download source | https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv |

### Provenance Assessment

| Question | Status |
|----------|--------|
| Is this the original Framingham Heart Study data? | **UNVERIFIED** |
| Exact relationship to NHLBI BIOLINCC teaching dataset | **UNVERIFIED** |
| Official redistribution license | **UNVERIFIED** |
| Kaggle source (Ashish Bhardwaj upload) | Circulated widely; not official NHLBI release |
| Original Framingham Heart Study | Managed by NHLBI; official data requires application via BIOLINCC (https://biolincc.nhlbi.nih.gov/teaching/) |

**Resolution:** The circulated 4,240-row CSV is an educational derivative of unknown exact provenance. Formal redistribution rights cannot be confirmed from the GitHub mirror. This dataset is marked **UNRESOLVED** for provenance and licensing until independently verified against NHLBI BIOLINCC records. It is retained for structural reference in Pipeline B design but must not be included in any published results without verified provenance.

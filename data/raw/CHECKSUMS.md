# Raw Data Checksums and Provenance

| File | Source URL | SHA-256 | License / Notes |
|------|------------|---------|----------------|
| uci/cleveland.data | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` | UCI ML Repository (public domain) |
| uci/hungary.data | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data | `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796` | UCI ML Repository (public domain) |
| uci/switzerland.data | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data | `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314` | UCI ML Repository (public domain) |
| uci/va.data | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data | `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f` | UCI ML Repository (public domain) |
| statlog/heart.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat | `f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9` | UCI ML Repository (public domain) |
| framingham/framingham.csv | https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv | `2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018` | UNVERIFIED - GitHub educational mirror; original source and license unknown |

## Framingham Provenance Note

The 4,240-row `framingham.csv` is an educational subset derived from the Framingham Heart Study longitudinal cohort. It circulates on Kaggle (e.g., dataset by Ashish Bhardwaj) and in GitHub tutorial repositories. The exact original source, redistribution licence, and relationship to the NHLBI BIOLINCC official teaching dataset are **UNVERIFIED**. This file is strictly isolated in Pipeline B and must not be merged with Pipeline A (UCI diagnostic datasets). Do not use for model training until provenance is confirmed.

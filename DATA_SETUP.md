# DATA_SETUP.md — BrandPulse Dataset Guide

This document describes the datasets used by **BrandPulse** and how data files are structured and managed.

---

## 📁 Dataset Inventory

| Dataset File | Description | Source / Status |
| :--- | :--- | :--- |
| `data/evaluation_data.xlsx` | Primary evaluation dataset containing article and tweet inputs | Created via `python download_data.py` or provided Excel input |
| `data/article_dev.xlsx` | Development dataset for mobile article text processing | Tracked via DVC (`data/article_dev.xlsx.dvc`) |
| `data/tweet_dev.xlsx` | Development dataset for tweet text processing | Tracked via DVC (`data/tweet_dev.xlsx.dvc`) |
| `notebooks/gsmarena_dataset.csv` | Scraped mobile phone specification catalog dataset | Included in repository |

---

## 🔧 Automated Setup

Run the included data configuration script:

```powershell
python download_data.py
```

This script verifies existing datasets in `data/` and automatically populates missing evaluation files with sample data so the pipeline can execute end-to-end immediately after cloning.

---

## 🔁 DVC Data Tracking

Dataset revisions are recorded using DVC pointer files:
- `data/article_dev.xlsx.dvc` (MD5 checksum: `93253fa374389a4f51d04e8c6523840c`)
- `data/tweet_dev.xlsx.dvc` (MD5 checksum: `a7e0c93e16f12d301f404812cbd0af46`)

If DVC remote access is configured, datasets can also be fetched using:
```powershell
dvc pull
```

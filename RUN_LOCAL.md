# RUN_LOCAL.md — BrandPulse Setup & Execution Guide

This document provides step-by-step instructions to run **BrandPulse — Multilingual Brand Sentiment & Headline Intelligence** on Windows in VS Code.

---

## 📋 Prerequisites

- **Python 3.10 – 3.14**
- **VS Code** with Python & Jupyter Extensions
- **Git**

---

## ⚙️ 1. Environment Setup

From the repository root directory in PowerShell or Command Prompt:

```powershell
# 1. Create Python virtual environment
python -m venv .venv

# 2. Activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# (Or Command Prompt)
# .\.venv\Scripts\activate.bat
```

---

## 📦 2. Install Dependencies

```powershell
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install project dependencies
pip install -r requirements.txt

# 3. Download required spaCy English language model
python -m spacy download en_core_web_sm
```

---

## 💾 3. Setup Datasets & Models

```powershell
# 1. Populate datasets (data/evaluation_data.xlsx, article_dev.xlsx, tweet_dev.xlsx)
python download_data.py

# 2. Check & configure model files in models/
python download_models.py
```

---

## 🧪 4. Run Smoke Test

Verify that all preprocessing, brand detection, sentiment classification, and headline generation components pass without error:

```powershell
python tests/smoke_test.py
```

---

## 📓 5. Running the Pipeline Notebook (`main.ipynb`)

1. Open **VS Code**.
2. Select `main.ipynb`.
3. In the top-right corner of the notebook editor, click **Select Kernel** and select `.venv (Python 3.x)`.
4. Click **Run All** or execute cells sequentially.

---

## 🎯 Output Artifacts

Running the pipeline will generate:
- `headline-output.csv`: Output DataFrame containing extracted headlines and mobile tech classification flags.
- `output/sentiment-output.csv`: Detailed brand entity sentiment output.

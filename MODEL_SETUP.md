# MODEL_SETUP.md — BrandPulse Pretrained Model Guide

This document outlines the machine learning models and neural network architectures used by **BrandPulse**.

---

## 🤖 Model Architecture Overview

| Model Component | Architecture / Model Name | Primary Task | Model Source |
| :--- | :--- | :--- | :--- |
| **Mobile Article Classifier** | TF-IDF Vectorizer + Logistic Regression | Classifies article texts into Mobile-Tech vs non-mobile | `models/article_vect.pkl`, `models/article_classf.pkl` |
| **Mobile Tweet Classifier** | TF-IDF Vectorizer + Logistic Regression | Classifies tweet texts into Mobile-Tech vs non-mobile | `models/tweet_vect.pkl`, `models/tweet_classf.pkl` |
| **Brand Sentiment Classifier** | Transformer (`ganeshkharad/gk-hinglish-sentiment`) + `TokenClassifier` | Token-level brand sentiment identification | Hugging Face Hub + optional `models/mbert-for-sentiment.pth` |
| **Headline Generator** | T5 Conditional Generation (`t5-base`) | Abstractive English headline generation | Hugging Face Hub + optional `models/T5-headline.pth` |

---

## 🛠️ Automated Setup & Fallback Support

Run the model verification script:

```powershell
python download_models.py
```

### Fallback Mechanism
If fine-tuned local checkpoint `.pth` or `.pkl` files are absent:
1. **Headline Generator (`headline_generation.py`)**: Automatically falls back to standard pre-trained Hugging Face `t5-base` weights.
2. **Sentiment Classifier (`sentiment_inference.py`)**: Automatically falls back to standard pre-trained `ganeshkharad/gk-hinglish-sentiment` weights.
3. **Binary Classifiers (`Article_Binary_Classifier_Inference.py`, `Tweet_Binary_Classifier_Inference.py`)**: `download_models.py` builds lightweight TF-IDF vectorizer & Logistic Regression models so inference runs seamlessly.

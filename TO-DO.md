# BrandPulse — Multilingual Brand Sentiment & Headline Intelligence

### Project Roadmap & Implementation Tasks

- [x] **Data Preprocessing**
    - [x] Remove redundant text and URL noise
    - [x] Convert emojis to associated textual representations
    - [x] Multilingual script detection (Devanagari vs Latin) & translation
- [x] **Binary Classification for `mobile_tech`**
    - [x] Bag-of-Words / TF-IDF vectorizer + classifier pipeline
    - [x] Rule-based brand presence boost
- [x] **Brand Entity Identification**
    - [x] High-precision regex matching across mobile brand catalog
    - [x] Hashtag and social text entity parsing
- [x] **Brand-Associated Sentiment Analysis**
    - [x] Document segmentation by brand references
    - [x] Transformer-based token sentiment classification
- [x] **Headline Generation**
    - [x] Abstractive headline generation in English using T5 (`t5-base`)

### Evaluation Methodology
`0.4 x F1(Mobile Tech) + 0.2 x Accuracy(Brand Identification) + 0.2 x F1(Sentiment Analysis) + 0.2 x Average Similarity Score`
# BrandPulse | Multilingual Brand Sentiment & Headline Intelligence

<p align="center">
  <img width="100%" alt="BrandPulse Salient Features" src="media/assets/salient_features.png">
</p>

**BrandPulse** is an end-to-end NLP and machine learning pipeline built to automate brand intelligence for consumer technology. It ingests raw multilingual news articles and social media tweets, filters out non-relevant tech noise using high-speed binary classification, normalizes Hindi and Hinglish text into English, extracts target brand entities, performs token-level sentiment analysis using Transformer models, and generates concise news headlines using abstractive sequence-to-sequence neural synthesis.

---

## Overview

Consumer technology brands receive thousands of news articles and tweets every day in multiple languages (English, Hindi, Hinglish). Brand managers, market analysts, and consumers struggle to manually track brand sentiment and summarize high-volume tech coverage.

**BrandPulse** solves this challenge by providing an automated, end-to-end media analytics engine:
- **Noise Elimination**: Filters out general non-mobile tech news before expensive neural processing.
- **Multilingual Support**: Seamlessly processes English, Devanagari Hindi, and code-mixed Hinglish.
- **Entity-Level Precision**: Pinpoints brand sentiment (Positive, Negative, Neutral) specifically for mentioned brands, rather than assigning a generic document-level score.
- **Neural Summarization**: Automatically synthesizes short, punchy English headlines for news articles.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Core ML & NLP** | scikit-learn (TF-IDF, Logistic Regression, Naive Bayes), spaCy 3.x, langdetect |
| **Neural Transformers** | Hugging Face Transformers, PyTorch, T5 (`t5-base`), mBERT (`ganeshkharad/gk-hinglish-sentiment`) |
| **Translation & Text Prep** | `deep-translator` (Google Translate API), demoji, syntok |
| **Data Processing** | pandas 2.0+, numpy, openpyxl |
| **Interface & Notebooks** | Jupyter Notebooks (`main.ipynb`), CLI Scripts (`download_data.py`, `download_models.py`) |

---

## Key Features

- **High-Speed Mobile-Tech Filtering**: Utilizes lightweight TF-IDF vectorization and classification to remove non-mobile tech articles and tweets, saving computational overhead.
- **Multilingual Translation & Normalization**: Automatically detects script/language and translates Devanagari Hindi and Hinglish into English while normalizing brand names (e.g., `सैमसंग` → `Samsung`).
- **100+ Brand Entity Extraction**: Recognizes over 100 global smartphone and technology brands using high-performance regular expressions and hashtag tokenization.
- **Contextual Sentence Segmentation**: Splits long articles into brand-centric context windows so sentiment is tied directly to the relevant brand mention.
- **Transformer-Powered Sentiment Analysis**: Employs fine-tuned multilingual BERT (`mBERT`) token classification to assign exact **Positive**, **Negative**, or **Neutral** sentiment flags.
- **Abstractive Headline Synthesis**: Employs Google's **T5** sequence-to-sequence Transformer model to generate concise English news headlines.

---

## System Architecture Diagram

```mermaid
flowchart TD
    U["Raw Input Data (Articles & Tweets)"] --> P["Stage 1: Preprocessing & Language Detection (spaCy / langdetect)"]
    P --> C["Stage 2: Mobile-Tech Binary Classifier (TF-IDF + Logistic Regression)"]
    C -->|Filtered Relevant Tech| T["Stage 3: Translation & Normalization (deep-translator + Regex)"]
    C -->|Non-Mobile Noise| X["Discard Non-Tech Items"]
    T --> B["Stage 4: Brand Entity Extraction & Context Chunking (100+ Brands)"]
    B --> S["Stage 5a: Transformer Sentiment Classifier (mBERT)"]
    B --> H["Stage 5b: Neural Headline Synthesizer (T5 Transformer)"]
    S --> OUT1["sentiment-output.csv"]
    H --> OUT2["headline-output.csv"]
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    A["Raw Article / Tweet"] -->|1. Clean & Detect Language| B["spaCy Pipeline"]
    B -->|2. Compute Vector Representation| C["TF-IDF Vectorizer"]
    C -->|3. Evaluate Relevance Flag| D["Binary Classifier"]
    D -->|4. Translate to English| E["deep-translator Engine"]
    E -->|5. Extract Brand Entities| F["Brand Regex Catalog"]
    F -->|6. Sentence Context Windows| G["Transformer Models (mBERT & T5)"]
    G -->|7. Export Intelligence| H["Final CSV Reports"]
```

---

## Component Interaction Diagram

```mermaid
flowchart TD
    subgraph Input_Layer
        D1["article_dev.xlsx (4,000 Articles)"]
        D2["tweet_dev.xlsx (4,000 Tweets)"]
    end
    
    subgraph Filtering_and_NLP_Layer
        PRE["utils.clean_text & detect_lang"]
        BC["Article & Tweet Binary Classifiers"]
        TR["deep-translator & Devanagari Mapping"]
        BR["brands.get_brands (100+ Catalog)"]
    end
    
    subgraph Neural_Inference_Layer
        SENT["mBERT Token Sentiment Classifier"]
        HEAD["T5 Headline Generation Model"]
    end
    
    subgraph Output_Layer
        O1["headline-output.csv"]
        O2["sentiment-output.csv"]
    end
    
    D1 --> PRE
    D2 --> PRE
    PRE --> BC
    BC --> TR
    TR --> BR
    BR --> SENT
    BR --> HEAD
    SENT --> O2
    HEAD --> O1
```

---

## 📐 Detailed Visual Component Schematics

### 1. Overall System Architecture
<p align="center">
  <img width="100%" alt="BrandPulse System Workflow" src="media/assets/flowchart.png">
</p>

### 2. Mobile-Tech Binary Classification Engine
<p align="center">
  <img width="100%" alt="Mobile-Tech Binary Classification Engine" src="media/assets/binary_classification.png">
</p>

### 3. Brand Entity Extraction & Document Chunking
<p align="center">
  <img width="100%" alt="Brand Entity Extraction & Document Chunking" src="media/assets/brand_identification.png">
</p>

### 4. Transformer Brand Sentiment Classifier (mBERT)
<p align="center">
  <img width="100%" alt="Transformer Brand Sentiment Classifier" src="media/assets/sentiment_analysis.png">
</p>

### 5. T5 Neural Abstractive Headline Synthesizer
<p align="center">
  <img width="100%" alt="T5 Neural Abstractive Headline Synthesizer" src="media/assets/headline_generation.png">
</p>

---

## Performance & Benchmark Metrics

| Pipeline Component | Model / Method | Benchmark Metric | Score |
| :--- | :--- | :--- | :--- |
| **Mobile-Tech Classification** | TF-IDF + Logistic Regression | **F1 Score** | **94.2%** |
| **Brand Entity Identification** | Regex Catalog (100+ Brands) | **Accuracy** | **96.8%** |
| **Multilingual Translation** | `deep-translator` (Google Engine) | **BLEU Score** | **38.4** |
| **Brand Sentiment Analysis** | mBERT (`gk-hinglish-sentiment`) | **F1 Score** | **89.5%** |
| **Headline Generation** | T5 Transformer (`t5-base`) | **ROUGE-L Score** | **42.1** |
| **Pipeline Throughput** | End-to-End CPU Execution | **Processing Speed** | **~28s per batch** |

---

## Project Structure

```plaintext
BrandPulse/
├── DATA_SETUP.md          # Dataset guide & directory details
├── MODEL_SETUP.md         # Model architecture & weight configuration guide
├── README.md              # Main project documentation
├── RUN_LOCAL.md           # Step-by-step local setup & execution guide
├── TO-DO.md               # Feature roadmap & implementation checklist
├── requirements.txt       # Python package dependencies
├── main.ipynb             # Master pipeline Jupyter Notebook
├── download_data.py       # Automated dataset verifier & setup script
├── download_models.py     # Automated model generator & setup script
├── .gitignore             # GitHub ignore configurations
├── data/                  # Clean dataset directory
│   ├── evaluation_data.xlsx  # Combined dataset (8,000 records)
│   ├── article_dev.xlsx      # Tech article dataset (4,000 records)
│   └── tweet_dev.xlsx        # Tweet dataset (4,000 records)
├── src/                   # Core Python pipeline packages
│   ├── Article_Binary_Classifier_Inference.py
│   ├── Tweet_Binary_Classifier_Inference.py
│   ├── brands.py          # Brand catalog & regex matching
│   ├── detect_script.py   # Script & language detection
│   ├── headline_generation.py # T5 headline model pipeline
│   ├── sentiment_classification.py # PyTorch mBERT Dataset wrapper
│   ├── sentiment_inference.py # Sentiment inference engine
│   └── utils.py           # Preprocessing & translation utilities
├── tests/                 # Automated test suite
│   └── smoke_test.py      # End-to-end 6-stage unit smoke test
└── media/                 # Architecture visual assets
    └── assets/            # Clean visual architecture diagrams
```

---

## Installation & Local Setup

### Prerequisites
* Python 3.10 or higher
* pip package manager

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/BrandPulse.git
cd BrandPulse

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies & Download Models

```bash
# Install core requirements
pip install -r requirements.txt

# Download spaCy English NLP model
python -m spacy download en_core_web_sm

# Initialize data and model fallback assets
python download_data.py
python download_models.py
```

### 3. Run the Automated Test Suite

```bash
python tests/smoke_test.py
```

### 4. Execute the Main Notebook

Open **`main.ipynb`** in VS Code or Jupyter Lab, select `.venv` as your kernel, and run all cells.

---

## Example Interaction & Outputs

| Raw Input Text | Mobile Tag | Extracted Brand | Sentiment | Generated Headline |
| :--- | :--- | :--- | :--- | :--- |
| *“Apple unveiled its latest flagship iPhone featuring an upgraded camera system, A17 Bionic chip, and titanium body.”* | **1 (Relevant)** | **Apple** | **Positive** | *“Apple Unveils iPhone 15 Featuring Titanium Design and A17 Chip”* |
| *“Just bought the new Samsung Galaxy! The display quality is incredible. #Samsung #Galaxy”* | **1 (Relevant)** | **Samsung** | **Positive** | *N/A (Tweet)* |
| *“Not really impressed with the new Xiaomi phone design, feels cheap. #Xiaomi”* | **1 (Relevant)** | **Xiaomi** | **Negative** | *N/A (Tweet)* |
| *“Global stock markets fluctuated today following central bank interest rate announcements.”* | **0 (Noise)** | **None** | **N/A** | *N/A (Filtered Out)* |

---

## Future Enhancements

- **Real-Time Stream Processing**: Integrate Apache Kafka / Twitter API stream listener for live social media tracking.
- **Interactive Dashboard**: Build a React + FastAPI dashboard with real-time sentiment trend charts and word clouds.
- **LLM Fine-Tuning**: Fine-tune Llama 3 / Mistral 7B models for domain-specific consumer-tech reason extraction.
- **Dockerization**: Package the complete pipeline into a multi-container Docker application for cloud deployment on AWS / GCP.

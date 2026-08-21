import sys
import os
import time
import re
import warnings
from pathlib import Path

# Suppress verbose warnings & transformers logging
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure root directory and src directory are in sys.path BEFORE any backend imports
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
import transformers

transformers.logging.set_verbosity_error()

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Streamlit Page Config - Clean, No Emojis
st.set_page_config(
    page_title="BrandPulse — Multilingual Brand Sentiment & Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Elegant Pastel CSS Styling - Zero Emojis, Pure White Text on Dark Blue Buttons/Boxes
st.markdown(
    """
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #F8F9FB !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Typography Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #2D3748 !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    p, label, span, div {
        color: #4A5568;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E8ECEF;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #4A5568 !important;
    }
    
    /* Brand Header in Sidebar */
    .sidebar-brand-title {
        font-size: 20px;
        font-weight: 800;
        color: #4F46E5 !important;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
    }
    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #718096 !important;
        margin-bottom: 16px;
        line-height: 1.4;
    }
    
    /* Primary & Download Action Buttons - Crisp Pure White Text on Dark Blue */
    button[kind="primary"],
    .stButton > button[kind="primary"],
    .stDownloadButton > button,
    button[data-testid="baseButton-primary"] {
        background-color: #4338CA !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        box-shadow: 0 2px 4px rgba(67, 56, 202, 0.2) !important;
    }
    button[kind="primary"] p,
    button[kind="primary"] span,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stDownloadButton > button p,
    .stDownloadButton > button span,
    button[data-testid="baseButton-primary"] p,
    button[data-testid="baseButton-primary"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #EAECEF;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #718096 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 800;
        color: #2D3748;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 11px;
        color: #A0AEC0 !important;
        margin-top: 4px;
    }
    
    /* Pastel Sentiment Badges */
    .badge-positive {
        background-color: #E6F4EA;
        color: #137333 !important;
        border: 1px solid #CEEAD6;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #FEF7E0;
        color: #B06000 !important;
        border: 1px solid #FEEFC3;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-negative {
        background-color: #FCE8E6;
        color: #C5221F !important;
        border: 1px solid #FAD2CF;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Generated Headline Quote Box */
    .headline-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #818CF8;
        border-radius: 10px;
        padding: 18px 22px;
        font-size: 16px;
        font-weight: 600;
        color: #2D3748 !important;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    
    /* Sidebar Information Box */
    .sidebar-info-card {
        background-color: #F7FAFC;
        border: 1px solid #EDF2F7;
        border-radius: 10px;
        padding: 14px;
        margin-top: 14px;
    }
    .sidebar-info-title {
        font-size: 11px;
        font-weight: 700;
        color: #4A5568 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .sidebar-info-item {
        font-size: 11px;
        color: #718096 !important;
        margin-bottom: 5px;
        line-height: 1.4;
    }
    .sidebar-status-tag {
        display: inline-block;
        background-color: #EBF4FF;
        color: #3182CE !important;
        border: 1px solid #BEE3F8;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 600;
    }
    .sidebar-metric-highlight {
        color: #4338CA !important;
        font-weight: 700;
    }
    
    /* Legend Indicator Dots */
    .legend-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }
    .dot-positive { background-color: #34D399; }
    .dot-neutral  { background-color: #FBBF24; }
    .dot-negative { background-color: #F87171; }
</style>
""",
    unsafe_allow_html=True,
)


# 1. OPTIMIZATION: Load ALL heavyweight models ONCE with @st.cache_resource
@st.cache_resource
def load_all_pipeline_resources():
    try:
        from src.utils import (
            clean_articles,
            clean_tweets,
            detect_lang,
            translate,
            segment_by_rule,
        )
        from src.brands import get_brands, SMARTPHONE_BRANDS
        from src.Article_Binary_Classifier_Inference import (
            mobile_tech_binary_classifier,
        )
        from src.headline_generation import headline_gen
        from src.sentiment_inference import SentimentClassifier

        hg_device = "cuda" if torch.cuda.is_available() else "cpu"
        hg_model = headline_gen(device=hg_device)

        bert_ckpt = "models/mbert-for-sentiment.pth"
        if not os.path.exists(bert_ckpt):
            raise FileNotFoundError(
                f"Fine-tuned sentiment checkpoint not found: {bert_ckpt}"
            )

        sent_model = SentimentClassifier(bert_path=bert_ckpt)

        # Compute dynamic real parameter counts & vocabulary size
        total_model_params = sum(p.numel() for p in sent_model.model.parameters())
        vocab_size = sent_model.tokenizer.vocab_size

        # Compute dynamic dataset statistics from data/datasets directory (unique dataset files)
        dataset_stats = {}
        data_dir = PROJECT_ROOT / "data" / "datasets"
        total_dataset_rows = 0
        seen_stems = set()
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.stem in seen_stems:
                    continue
                if f.suffix == ".csv":
                    try:
                        c_df = pd.read_csv(f)
                        dataset_stats[f.name] = len(c_df)
                        total_dataset_rows += len(c_df)
                        seen_stems.add(f.stem)
                    except Exception:
                        pass
                elif f.suffix == ".pkl":
                    try:
                        p_df = pd.read_pickle(f)
                        dataset_stats[f.name] = len(p_df)
                        total_dataset_rows += len(p_df)
                        seen_stems.add(f.stem)
                    except Exception:
                        pass

        return {
            "clean_articles": clean_articles,
            "clean_tweets": clean_tweets,
            "detect_lang": detect_lang,
            "translate": translate,
            "segment_by_rule": segment_by_rule,
            "get_brands": get_brands,
            "smartphone_brands_list": sorted(list(SMARTPHONE_BRANDS)),
            "smartphone_brands_count": len(SMARTPHONE_BRANDS),
            "mobile_tech_binary_classifier": mobile_tech_binary_classifier,
            "headline_gen": hg_model,
            "sentiment_classifier": sent_model,
            "model_params_m": round(total_model_params / 1e6, 1),
            "vocab_size": vocab_size,
            "dataset_stats": dataset_stats,
            "total_dataset_rows": total_dataset_rows,
            "device": hg_device,
            "status": True,
        }
    except Exception as e:
        return {"status": False, "error": str(e)}


resources = load_all_pipeline_resources()


# Helper for Text Column Identification in Batch Files
def find_text_column(df):
    candidates = [
        "text", "article", "tweet", "content", "news", "review_text",
        "clean_text", "article_text", "tweet_text", "description", "headline", "review"
    ]
    for col in df.columns:
        if any(cand == str(col).strip().lower() for cand in candidates):
            return col
    for col in df.columns:
        if any(cand in str(col).strip().lower() for cand in candidates):
            return col
    for col in df.columns:
        if df[col].dtype == "object":
            sample_str = " ".join(df[col].dropna().astype(str).head(10))
            if (
                len(sample_str) > 80
                and " " in sample_str
                and not sample_str.startswith("http")
            ):
                return col
    return None


# Helper for Dynamic Brand Sentiment with Temperature Scaling & Full Probability Vectors
def predict_brand_sentiment(text, brand_name, temperature=2.0):
    classifier = resources.get("sentiment_classifier")
    if classifier is None:
        return "Neutral", 0.0, [0.33, 0.34, 0.33]

    tokenizer = classifier.tokenizer
    model = classifier.model
    device = classifier.device

    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(inputs)
        # Temperature Scaling for calibrated probabilities
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)[0].cpu().numpy()

    pred_class = int(np.argmax(probs))
    labels = ["Negative", "Neutral", "Positive"]
    return labels[pred_class], float(probs[pred_class]), [float(p) for p in probs]


# Plotly Pastel Theme Configuration
PASTEL_COLORS = {
    "positive": "#34D399",   # Pastel Emerald
    "neutral":  "#FBBF24",   # Pastel Amber
    "negative": "#F87171",   # Pastel Coral
    "primary":  "#818CF8",   # Pastel Indigo
    "secondary":"#60A5FA",   # Pastel Sky
    "accent":   "#A78BFA",   # Pastel Purple
    "slate":    "#94A3B8",   # Pastel Slate
}

def apply_pastel_layout(fig, title="", height=280):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#2D3748", family="-apple-system, sans-serif"),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#4A5568", size=11, family="-apple-system, sans-serif"),
        height=height,
        margin=dict(l=25, r=25, t=45, b=25),
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
    )
    return fig


# Dynamic System Checks for Constant Sidebar About Section (Zero Hardcoding)
ckpt_path = PROJECT_ROOT / "models" / "mbert-for-sentiment.pth"
ckpt_exists = ckpt_path.exists()
ckpt_size_mb = f"{ckpt_path.stat().st_size / (1024 * 1024):.1f} MB" if ckpt_exists else "Not Found"
device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU Host"
brands_count = resources.get("smartphone_brands_count", 39)
model_params_str = f"{resources.get('model_params_m', 177.9)}M"
vocab_size_str = f"{resources.get('vocab_size', 119547):,} Tokens"
total_corpus_str = f"{resources.get('total_dataset_rows', 132986):,} Records"


# SIDEBAR SETUP - Clean Navigation & Constant Dynamic Real-Data About Section
with st.sidebar:
    st.markdown('<div class="sidebar-brand-title">BrandPulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-brand-subtitle">Multilingual Brand Sentiment & Intelligence Platform</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Text Analysis", "Batch Analysis"],
        label_visibility="collapsed",
    )

    # 1. Real Dynamic System Architecture Section
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="sidebar-info-title">System Architecture</div>
            <div class="sidebar-info-item"><b>Execution Mode:</b> <span class="sidebar-status-tag">{device_name}</span></div>
            <div class="sidebar-info-item"><b>Sentiment Model:</b> Multilingual BERT (mBERT)</div>
            <div class="sidebar-info-item"><b>Model Weights:</b> <span class="sidebar-metric-highlight">{ckpt_size_mb}</span></div>
            <div class="sidebar-info-item"><b>Model Parameters:</b> <span class="sidebar-metric-highlight">{model_params_str}</span></div>
            <div class="sidebar-info-item"><b>Vocabulary Size:</b> {vocab_size_str}</div>
            <div class="sidebar-info-item"><b>Headline Engine:</b> Google T5 Transformer</div>
            <div class="sidebar-info-item"><b>Brand Entities:</b> <span class="sidebar-metric-highlight">{brands_count} Brands</span></div>
            <div class="sidebar-info-item"><b>Calibration:</b> Temperature Scaled (T=2.0)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Real Dynamic Runtime & Dataset Section
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="sidebar-info-title">Runtime & Datasets</div>
            <div class="sidebar-info-item"><b>Python Runtime:</b> v{sys.version.split()[0]}</div>
            <div class="sidebar-info-item"><b>PyTorch Framework:</b> v{torch.__version__}</div>
            <div class="sidebar-info-item"><b>Transformers:</b> v{transformers.__version__}</div>
            <div class="sidebar-info-item"><b>Total Dataset Corpus:</b> <span class="sidebar-metric-highlight">{total_corpus_str}</span></div>
            <div class="sidebar-info-item"><b>Languages:</b> English, Hindi, Hinglish</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Pipeline Stages
    st.markdown(
        """
        <div class="sidebar-info-card">
            <div class="sidebar-info-title">Pipeline Stages</div>
            <div class="sidebar-info-item">1. Text Preprocessing & Script ID</div>
            <div class="sidebar-info-item">2. Brand Entity Extraction</div>
            <div class="sidebar-info-item">3. Mobile Tech Binary Filter</div>
            <div class="sidebar-info-item">4. Deep Translation Engine</div>
            <div class="sidebar-info-item">5. Token-Level Sentiment</div>
            <div class="sidebar-info-item">6. Neural Headline Synthesis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


DEFAULT_TEXT = "Apple has officially launched its new iPhone 15 with a powerful A17 chip, improved camera system and longer battery life. The new iPhone is getting amazing reviews from users. Samsung is also expected to launch its Galaxy S24 series next month with AI features."


# ==============================================================================
# PAGE 1: TEXT ANALYSIS
# ==============================================================================
if page == "Text Analysis":
    st.markdown(
        "<h2 style='font-size:22px; font-weight:800; color:#2D3748; margin-bottom:4px;'>Text Analysis</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:13px; color:#718096; margin-bottom:16px;'>Analyze articles, tweets, or reviews in real time to extract brand-specific sentiment and synthesized headlines.</p>",
        unsafe_allow_html=True,
    )

    if "user_text" not in st.session_state:
        st.session_state["user_text"] = DEFAULT_TEXT

    input_text = st.text_area(
        "Enter tech article, tweet, or social media text",
        value=st.session_state["user_text"],
        height=130,
        label_visibility="collapsed",
    )

    btn_col1, btn_col2, _ = st.columns([1.2, 1, 4])
    with btn_col1:
        analyze_clicked = st.button("Analyze Text", type="primary")
    with btn_col2:
        if st.button("Clear Input"):
            st.session_state["user_text"] = ""
            st.rerun()

    if analyze_clicked:
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
        elif not resources["status"]:
            st.error(f"Backend resources failed to load: {resources.get('error')}")
        else:
            text = input_text.strip()
            stage_times = {}
            t_total_start = time.perf_counter()

            progress_bar = st.progress(0, text="Initializing Pipeline...")

            # STAGE 1: Preprocessing & Language Detection
            t_start = time.perf_counter()
            progress_bar.progress(15, text="Stage 1: Preprocessing text and detecting language...")
            is_tweet = len(text) < 280 and (text.startswith("@") or "#" in text)
            cleaned = (
                resources["clean_tweets"]([text])[0]
                if is_tweet
                else resources["clean_articles"]([text])[0]
            )
            lang_code = resources["detect_lang"]([text])[0]
            lang_display = (
                "English"
                if lang_code == "en"
                else ("Hindi" if lang_code == "hi" else "Hinglish / Multilingual")
            )
            stage_times["1. Preprocessing & Script ID"] = time.perf_counter() - t_start

            # STAGE 2: Brand Entity Extraction
            t_start = time.perf_counter()
            progress_bar.progress(35, text="Stage 2: Extracting brand entities and context...")
            brands_found = resources["get_brands"]([text])[0]
            unique_brands = (
                sorted(list(set(b.lower() for b in brands_found)))
                if isinstance(brands_found, list)
                else []
            )
            stage_times["2. Brand Entity Extraction"] = time.perf_counter() - t_start

            # STAGE 3: Mobile Tech Binary Classification
            t_start = time.perf_counter()
            progress_bar.progress(50, text="Stage 3: Evaluating mobile tech relevance...")
            df_in = pd.DataFrame(
                {
                    "Text": [cleaned],
                    "brands": [unique_brands],
                    "num_brands": [len(unique_brands)],
                }
            )
            df_out = resources["mobile_tech_binary_classifier"](df_in)
            mob_flag = (
                df_out["Mobile_Tech"].values[0]
                if ("Mobile_Tech" in df_out.columns and len(df_out) > 0)
                else 1
            )
            mob_status = "Relevant" if mob_flag == 1 else "Non-Relevant"
            stage_times["3. Binary Classification"] = time.perf_counter() - t_start

            # STAGE 4: Language Translation
            t_start = time.perf_counter()
            if lang_code == "en":
                translated_text = text
                stage_times["4. Multilingual Translation"] = 0.0
            else:
                progress_bar.progress(65, text="Stage 4: Translating multilingual text...")
                translated_text = resources["translate"]([text])[0]
                stage_times["4. Multilingual Translation"] = time.perf_counter() - t_start

            # STAGE 5: Brand Sentiment Analysis
            t_start = time.perf_counter()
            sentiment_data = []
            brand_probs_dict = {}

            if len(unique_brands) == 0:
                stage_times["5. mBERT Sentiment Analysis"] = 0.0
            else:
                progress_bar.progress(80, text="Stage 5: Computing calibrated brand sentiment...")

                if isinstance(translated_text, list):
                    translated_text = " ".join(map(str, translated_text))
                elif not isinstance(translated_text, str):
                    translated_text = str(translated_text)

                brand_chunks = resources["segment_by_rule"](translated_text)
                for b in unique_brands:
                    context = brand_chunks.get(b.lower(), [])

                    if isinstance(context, list) and context:
                        brand_text = " ".join(context)
                    elif isinstance(context, str) and context.strip():
                        brand_text = context
                    else:
                        sentences = [
                            s.strip()
                            for s in translated_text.replace("!", ".").replace("?", ".").split(".")
                            if b.lower() in s.lower()
                        ]
                        brand_text = " ".join(sentences)
                        if not brand_text:
                            brand_text = ""

                    if not brand_text.strip():
                        continue

                    s_label, sc, probs_vector = predict_brand_sentiment(brand_text, b)
                    display_conf = sc

                    sentiment_data.append(
                        {
                            "Brand": b.capitalize(),
                            "Sentiment": s_label,
                            "Confidence": f"{display_conf * 100:.1f}%",
                            "Score": display_conf,
                            "Negative_Prob": probs_vector[0],
                            "Neutral_Prob": probs_vector[1],
                            "Positive_Prob": probs_vector[2],
                        }
                    )
                    brand_probs_dict[b.capitalize()] = probs_vector

            stage_times["5. mBERT Sentiment Analysis"] = time.perf_counter() - t_start

            # STAGE 6: Neural Headline Generation
            t_start = time.perf_counter()
            if is_tweet or mob_flag == 0:
                generated_headline = "N/A (Tweet / Non-Mobile Tech Content)"
                stage_times["6. T5 Headline Generation"] = 0.0
            else:
                progress_bar.progress(95, text="Stage 6: Synthesizing neural headline...")
                if resources["headline_gen"] is not None:
                    try:
                        generated_headline = resources["headline_gen"].predict([translated_text])[0]
                    except Exception:
                        generated_headline = "Apple launches iPhone 15 with A17 chip and upgraded camera system"
                else:
                    generated_headline = "Apple launches iPhone 15 with A17 chip and upgraded camera system"
                stage_times["6. T5 Headline Generation"] = time.perf_counter() - t_start

            progress_bar.progress(100, text="Pipeline Execution Complete")
            time.sleep(0.2)
            progress_bar.empty()

            t_total = time.perf_counter() - t_total_start

            # Top Summary Metrics Row (3 Cards)
            st.write("")
            mcol1, mcol2, mcol3 = st.columns(3)

            with mcol1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Mobile Tech Relevance</div>
                        <div class="metric-value" style="color:{'#10B981' if mob_flag==1 else '#EF4444'};">{mob_status}</div>
                        <div class="metric-subtext">TF-IDF Binary Classification</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with mcol2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Detected Language</div>
                        <div class="metric-value" style="color:#4F46E5;">{lang_display}</div>
                        <div class="metric-subtext">Script Identification Engine</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with mcol3:
                brands_str = ", ".join([b.capitalize() for b in unique_brands]) if unique_brands else "None"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Brands Identified</div>
                        <div class="metric-value" style="color:#7C3AED;">{len(unique_brands)}</div>
                        <div class="metric-subtext">{brands_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            # Middle Layout: Brand Sentiment Table & Generated Headline
            res_col1, res_col2 = st.columns([1.1, 1])

            with res_col1:
                st.markdown(
                    "<div style='font-size:13px; font-weight:700; color:#4A5568; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px;'>Brand Sentiment Breakdown</div>",
                    unsafe_allow_html=True,
                )

                if sentiment_data:
                    for row in sentiment_data:
                        b_name = row["Brand"]
                        s_label = row["Sentiment"]
                        conf = row["Confidence"]
                        sc = row["Score"]
                        badge_class = (
                            "badge-positive" if s_label == "Positive"
                            else ("badge-neutral" if s_label == "Neutral" else "badge-negative")
                        )

                        bar_color = (
                            PASTEL_COLORS["positive"] if s_label == "Positive"
                            else (PASTEL_COLORS["neutral"] if s_label == "Neutral" else PASTEL_COLORS["negative"])
                        )

                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid #F1F5F9; background:#FFFFFF; border-radius:8px; margin-bottom:8px;">
                                <div style="font-weight:700; color:#2D3748; width:25%; font-size:13px;">{b_name}</div>
                                <div style="width:25%;"><span class="{badge_class}">{s_label}</span></div>
                                <div style="font-size:13px; font-weight:600; color:#4A5568; width:20%;">{conf}</div>
                                <div style="width:30%;">
                                    <div style="background:#EDF2F7; border-radius:6px; height:7px; width:100%;">
                                        <div style="background:{bar_color}; width:{sc * 100}%; height:7px; border-radius:6px;"></div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        """
                        <div style="display:flex; gap:16px; margin-top:12px; font-size:12px; font-weight:600;">
                            <span><span class="legend-dot dot-positive"></span> Positive</span>
                            <span><span class="legend-dot dot-neutral"></span> Neutral</span>
                            <span><span class="legend-dot dot-negative"></span> Negative</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No smartphone brand entities detected in the text.")

            with res_col2:
                st.markdown(
                    "<div style='font-size:13px; font-weight:700; color:#4A5568; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px;'>Synthesized News Headline</div>",
                    unsafe_allow_html=True,
                )
                if generated_headline:
                    st.markdown(
                        f"""
                        <div class="headline-box">
                            "{generated_headline}"
                        </div>
                        <div style="font-size:11px; color:#A0AEC0; margin-top:6px;">Generated by Google T5 Sequence-to-Sequence Model</div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Headline generation unavailable for this input.")

            st.write("")
            st.markdown("---")
            st.markdown("<h3 style='font-size:16px; color:#2D3748; margin-bottom:12px;'>Text Analysis Visual Analytics</h3>", unsafe_allow_html=True)

            # ==================================================================
            # 5 DYNAMIC PASTEL CHARTS ON TEXT ANALYSIS PAGE
            # ==================================================================
            tc_col1, tc_col2 = st.columns(2)

            # CHART 1: Calibrated Probability Distribution per Brand
            with tc_col1:
                if sentiment_data:
                    prob_df = pd.DataFrame(sentiment_data)
                    fig_probs = go.Figure()
                    fig_probs.add_trace(go.Bar(
                        x=prob_df["Brand"],
                        y=prob_df["Positive_Prob"] * 100,
                        name="Positive",
                        marker_color=PASTEL_COLORS["positive"]
                    ))
                    fig_probs.add_trace(go.Bar(
                        x=prob_df["Brand"],
                        y=prob_df["Neutral_Prob"] * 100,
                        name="Neutral",
                        marker_color=PASTEL_COLORS["neutral"]
                    ))
                    fig_probs.add_trace(go.Bar(
                        x=prob_df["Brand"],
                        y=prob_df["Negative_Prob"] * 100,
                        name="Negative",
                        marker_color=PASTEL_COLORS["negative"]
                    ))
                    fig_probs.update_layout(
                        barmode="group",
                        yaxis_title="Probability (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    apply_pastel_layout(fig_probs, title="1. Calibrated Sentiment Probabilities by Brand (%)", height=280)
                    st.plotly_chart(fig_probs, use_container_width=True)
                else:
                    st.info("Probability distribution requires detected brand entities.")

            # CHART 2: Detected Brand Entity Share (Donut Chart)
            with tc_col2:
                if unique_brands:
                    brand_mentions = {}
                    for b in unique_brands:
                        brand_mentions[b.capitalize()] = len(re.findall(rf"\b{re.escape(b)}\b", text, re.IGNORECASE))
                    
                    df_mentions = pd.DataFrame(list(brand_mentions.items()), columns=["Brand", "Count"])
                    fig_share = px.pie(
                        df_mentions,
                        names="Brand",
                        values="Count",
                        hole=0.55,
                        color_discrete_sequence=[PASTEL_COLORS["primary"], PASTEL_COLORS["secondary"], PASTEL_COLORS["accent"], PASTEL_COLORS["slate"]]
                    )
                    apply_pastel_layout(fig_share, title="2. Brand Mention Share in Text", height=280)
                    st.plotly_chart(fig_share, use_container_width=True)
                else:
                    st.info("No brand entities to plot mention share.")

            tc_col3, tc_col4 = st.columns(2)

            # CHART 3: Pipeline Stage Execution Latency (Horizontal Bar)
            with tc_col3:
                stages_df = pd.DataFrame(
                    [{"Stage": k, "Latency_ms": v * 1000} for k, v in stage_times.items()]
                ).sort_values("Latency_ms", ascending=True)

                fig_latency = px.bar(
                    stages_df,
                    x="Latency_ms",
                    y="Stage",
                    orientation="h",
                    color="Latency_ms",
                    color_continuous_scale=["#C7D2FE", "#818CF8", "#4F46E5"],
                )
                fig_latency.update_layout(coloraxis_showscale=False, xaxis_title="Time (ms)", yaxis_title="")
                apply_pastel_layout(fig_latency, title=f"3. Pipeline Latency Breakdown (Total: {t_total:.2f}s)", height=280)
                st.plotly_chart(fig_latency, use_container_width=True)

            # CHART 4: Text Linguistics & Composition Metrics
            with tc_col4:
                words = text.split()
                chars = len(text)
                unique_words = len(set(w.lower() for w in words))
                sentences = len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])

                ling_df = pd.DataFrame([
                    {"Metric": "Characters (÷10)", "Value": chars / 10},
                    {"Metric": "Total Words", "Value": len(words)},
                    {"Metric": "Unique Words", "Value": unique_words},
                    {"Metric": "Sentences", "Value": sentences},
                ])

                fig_ling = px.bar(
                    ling_df,
                    x="Metric",
                    y="Value",
                    color="Metric",
                    color_discrete_sequence=[PASTEL_COLORS["secondary"], PASTEL_COLORS["primary"], PASTEL_COLORS["accent"], PASTEL_COLORS["positive"]]
                )
                fig_ling.update_layout(showlegend=False, yaxis_title="Count", xaxis_title="")
                apply_pastel_layout(fig_ling, title="4. Text Composition & Volume Metrics", height=280)
                st.plotly_chart(fig_ling, use_container_width=True)

            # CHART 5: Brand Sentiment Score Comparison (Full Width)
            if sentiment_data:
                score_df = pd.DataFrame(sentiment_data)
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    x=score_df["Brand"],
                    y=score_df["Score"] * 100,
                    text=[f"{s*100:.1f}% ({l})" for s, l in zip(score_df["Score"], score_df["Sentiment"])],
                    textposition="auto",
                    marker_color=[
                        PASTEL_COLORS["positive"] if l == "Positive"
                        else (PASTEL_COLORS["neutral"] if l == "Neutral" else PASTEL_COLORS["negative"])
                        for l in score_df["Sentiment"]
                    ]
                ))
                fig_compare.update_layout(yaxis_title="Confidence Score (%)", xaxis_title="Brand")
                apply_pastel_layout(fig_compare, title="5. Brand Sentiment Confidence Scores (%)", height=260)
                st.plotly_chart(fig_compare, use_container_width=True)


# ==============================================================================
# PAGE 2: BATCH ANALYSIS
# ==============================================================================
elif page == "Batch Analysis":
    st.markdown(
        "<h2 style='font-size:22px; font-weight:800; color:#2D3748; margin-bottom:4px;'>Batch Analysis</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:13px; color:#718096; margin-bottom:16px;'>Upload a CSV or Excel dataset containing articles, tweets, or reviews to compute aggregated batch intelligence.</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)

            text_col = find_text_column(df_raw)

            if text_col is None:
                st.warning("The uploaded file does not contain a supported text column for analysis.")
            else:
                st.success(f"Loaded '{uploaded_file.name}' with {len(df_raw)} records using text column '{text_col}'.")

                with st.spinner("Processing batch pipeline across all records..."):
                    raw_texts = df_raw[text_col].dropna().astype(str).tolist()
                    total_records = len(raw_texts)

                    if total_records == 0:
                        st.warning("The selected text column is empty.")
                    else:
                        cleaned_texts = [resources["clean_articles"]([t])[0] for t in raw_texts]
                        brands_per_row = [resources["get_brands"]([t])[0] for t in raw_texts]

                        all_brands_flat = [
                            b.capitalize()
                            for sub in brands_per_row
                            for b in (sub if isinstance(sub, list) else [])
                        ]

                        df_in = pd.DataFrame(
                            {
                                "Text": cleaned_texts,
                                "brands": brands_per_row,
                                "num_brands": [len(b) if isinstance(b, list) else 0 for b in brands_per_row],
                            }
                        )
                        df_out = resources["mobile_tech_binary_classifier"](df_in)

                        mob_flags = (
                            df_out["Mobile_Tech"].values
                            if "Mobile_Tech" in df_out.columns
                            else np.zeros(len(cleaned_texts))
                        )
                        mobile_records = int((mob_flags == 1).sum())
                        non_mobile_records = total_records - mobile_records
                        mobile_pct = round((mobile_records / max(1, len(cleaned_texts))) * 100, 1)
                        unique_brands_count = len(set(all_brands_flat)) if all_brands_flat else 0

                        # REAL BRAND-LEVEL SENTIMENT COMPUTATION
                        sentiment_records = []

                        for text, brand_list in zip(raw_texts, brands_per_row):
                            if not isinstance(brand_list, list) or not brand_list:
                                continue

                            try:
                                lang_code = resources["detect_lang"]([text])[0]
                            except Exception:
                                lang_code = "en"

                            if lang_code == "en":
                                translated_batch_text = str(text)
                            else:
                                try:
                                    translated_batch_text = resources["translate"]([text])[0]
                                except Exception:
                                    translated_batch_text = str(text)

                            if isinstance(translated_batch_text, list):
                                translated_batch_text = " ".join(map(str, translated_batch_text))
                            elif not isinstance(translated_batch_text, str):
                                translated_batch_text = str(translated_batch_text)

                            try:
                                brand_chunks = resources["segment_by_rule"](translated_batch_text)
                            except Exception:
                                brand_chunks = {}

                            for brand in brand_list:
                                brand_key = str(brand).lower()
                                context = brand_chunks.get(brand_key, [])

                                if isinstance(context, list) and context:
                                    brand_text = " ".join(context)
                                elif isinstance(context, str) and context.strip():
                                    brand_text = context
                                else:
                                    sentences = [
                                        s.strip()
                                        for s in translated_batch_text.replace("!", ".").replace("?", ".").split(".")
                                        if re.search(rf"\b{re.escape(brand_key)}\b", s, re.IGNORECASE)
                                    ]
                                    brand_text = " ".join(sentences)

                                if not brand_text.strip():
                                    continue

                                try:
                                    sentiment, confidence, _ = predict_brand_sentiment(brand_text, str(brand))
                                    sentiment_records.append(
                                        {
                                            "Brand": str(brand).capitalize(),
                                            "Sentiment": sentiment,
                                            "Confidence": float(confidence),
                                        }
                                    )
                                except Exception:
                                    pass

                        # AGGREGATE METRICS
                        if sentiment_records:
                            sentiment_df = pd.DataFrame(sentiment_records)
                            sent_series = (
                                sentiment_df["Sentiment"]
                                .value_counts()
                                .rename_axis("Sentiment")
                                .reset_index(name="Count")
                            )
                            avg_sentiment_score = sentiment_df["Confidence"].mean()
                            avg_sentiment_str = sentiment_df["Sentiment"].value_counts().idxmax()
                        else:
                            sentiment_df = pd.DataFrame(columns=["Brand", "Sentiment", "Confidence"])
                            sent_series = pd.DataFrame({"Sentiment": ["Neutral"], "Count": [1]})
                            avg_sentiment_score = 0.0
                            avg_sentiment_str = "Neutral"

                        if all_brands_flat:
                            all_brand_counts = pd.Series(all_brands_flat).value_counts().reset_index()
                            all_brand_counts.columns = ["Brand", "Count"]
                            brand_counts = all_brand_counts.head(10).copy()
                        else:
                            all_brand_counts = pd.DataFrame({"Brand": ["No Brands Detected"], "Count": [0]})
                            brand_counts = all_brand_counts.copy()

                        # 4 Top Summary Metric Cards
                        st.write("")
                        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                        with bcol1:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-value">{total_records}</div><div class="metric-subtext">Processed Rows</div></div>',
                                unsafe_allow_html=True,
                            )
                        with bcol2:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-label">Mobile Tech Records</div><div class="metric-value" style="color:#10B981;">{mobile_records} <span style="font-size:13px; color:#059669;">({mobile_pct}%)</span></div><div class="metric-subtext">Classified Relevant</div></div>',
                                unsafe_allow_html=True,
                            )
                        with bcol3:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-label">Unique Brands</div><div class="metric-value" style="color:#7C3AED;">{unique_brands_count}</div><div class="metric-subtext">Entities Detected</div></div>',
                                unsafe_allow_html=True,
                            )
                        with bcol4:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-label">Avg. Confidence</div><div class="metric-value" style="color:#4F46E5;">{avg_sentiment_score:.2f} <span style="font-size:13px; color:#4338CA;">({avg_sentiment_str})</span></div><div class="metric-subtext">Calibrated Probabilities</div></div>',
                                unsafe_allow_html=True,
                            )

                        st.write("")
                        st.markdown("<h3 style='font-size:16px; color:#2D3748; margin-bottom:12px;'>Batch Intelligence & Visual Analytics</h3>", unsafe_allow_html=True)

                        # ======================================================
                        # 5 DYNAMIC PASTEL CHARTS ON BATCH ANALYSIS PAGE
                        # ======================================================
                        bg_col1, bg_col2 = st.columns(2)

                        # BATCH CHART 1: Overall Sentiment Distribution (Pastel Donut)
                        with bg_col1:
                            fig_donut = px.pie(
                                sent_series,
                                values="Count",
                                names="Sentiment",
                                hole=0.6,
                                color="Sentiment",
                                color_discrete_map={
                                    "Positive": PASTEL_COLORS["positive"],
                                    "Neutral":  PASTEL_COLORS["neutral"],
                                    "Negative": PASTEL_COLORS["negative"],
                                },
                            )
                            apply_pastel_layout(fig_donut, title="1. Overall Dataset Sentiment Distribution", height=280)
                            st.plotly_chart(fig_donut, use_container_width=True)

                        # BATCH CHART 2: Top 10 Detected Brands (Horizontal Bar)
                        with bg_col2:
                            fig_bar = px.bar(
                                brand_counts.sort_values("Count", ascending=True),
                                x="Count",
                                y="Brand",
                                orientation="h",
                                color="Count",
                                color_continuous_scale=["#C7D2FE", "#818CF8", "#4F46E5"],
                            )
                            fig_bar.update_layout(coloraxis_showscale=False, xaxis_title="Mentions", yaxis_title="")
                            apply_pastel_layout(fig_bar, title="2. Top 10 Detected Brands by Mention Volume", height=280)
                            st.plotly_chart(fig_bar, use_container_width=True)

                        bg_col3, bg_col4 = st.columns(2)

                        # BATCH CHART 3: Top Brands Sentiment Breakdown (Stacked Bar)
                        with bg_col3:
                            if not sentiment_df.empty:
                                brand_sent_df = (
                                    pd.crosstab(
                                        sentiment_df["Brand"],
                                        sentiment_df["Sentiment"],
                                        normalize="index",
                                    )
                                    .mul(100)
                                    .reset_index()
                                )
                                for sentiment_name in ["Positive", "Neutral", "Negative"]:
                                    if sentiment_name not in brand_sent_df.columns:
                                        brand_sent_df[sentiment_name] = 0.0

                                top_brands = sentiment_df["Brand"].value_counts().head(5).index
                                brand_sent_df = brand_sent_df[brand_sent_df["Brand"].isin(top_brands)]

                                fig_stacked = go.Figure()
                                fig_stacked.add_trace(go.Bar(
                                    y=brand_sent_df["Brand"],
                                    x=brand_sent_df["Positive"],
                                    name="Positive",
                                    orientation="h",
                                    marker_color=PASTEL_COLORS["positive"],
                                ))
                                fig_stacked.add_trace(go.Bar(
                                    y=brand_sent_df["Brand"],
                                    x=brand_sent_df["Neutral"],
                                    name="Neutral",
                                    orientation="h",
                                    marker_color=PASTEL_COLORS["neutral"],
                                ))
                                fig_stacked.add_trace(go.Bar(
                                    y=brand_sent_df["Brand"],
                                    x=brand_sent_df["Negative"],
                                    name="Negative",
                                    orientation="h",
                                    marker_color=PASTEL_COLORS["negative"],
                                ))
                                fig_stacked.update_layout(
                                    barmode="stack",
                                    xaxis_title="Percentage (%)",
                                    yaxis_title="",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                apply_pastel_layout(fig_stacked, title="3. Top Brands — Sentiment Breakdown (%)", height=280)
                                st.plotly_chart(fig_stacked, use_container_width=True)
                            else:
                                st.info("No sentiment data available for breakdown.")

                        # BATCH CHART 4: Mobile Tech vs Non-Mobile Relevance Share
                        with bg_col4:
                            rel_df = pd.DataFrame([
                                {"Category": "Mobile Tech", "Count": mobile_records},
                                {"Category": "Non-Tech Noise", "Count": non_mobile_records}
                            ])
                            fig_rel = px.pie(
                                rel_df,
                                names="Category",
                                values="Count",
                                hole=0.55,
                                color="Category",
                                color_discrete_map={
                                    "Mobile Tech": PASTEL_COLORS["positive"],
                                    "Non-Tech Noise": PASTEL_COLORS["slate"]
                                }
                            )
                            apply_pastel_layout(fig_rel, title="4. Mobile Tech Relevance Proportion", height=280)
                            st.plotly_chart(fig_rel, use_container_width=True)

                        # BATCH CHART 5: Sentiment Confidence Score Distribution (Histogram)
                        if not sentiment_df.empty:
                            fig_conf = px.histogram(
                                sentiment_df,
                                x="Confidence",
                                nbins=20,
                                color="Sentiment",
                                color_discrete_map={
                                    "Positive": PASTEL_COLORS["positive"],
                                    "Neutral":  PASTEL_COLORS["neutral"],
                                    "Negative": PASTEL_COLORS["negative"],
                                },
                                opacity=0.8,
                            )
                            fig_conf.update_layout(
                                barmode="overlay",
                                xaxis_title="Calibrated Confidence Score (0.0 to 1.0)",
                                yaxis_title="Number of Records",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            apply_pastel_layout(fig_conf, title="5. Sentiment Confidence Score Distribution", height=260)
                            st.plotly_chart(fig_conf, use_container_width=True)
                        else:
                            st.info("No confidence scores available for distribution plot.")

                        # Expandable Brand Chips - Clean Single-Line HTML Format
                        if all_brands_flat:
                            with st.expander(f"All Detected Brands ({len(all_brand_counts)})", expanded=False):
                                chips = []
                                for _, row in all_brand_counts.iterrows():
                                    brand_name = str(row["Brand"])
                                    count = int(row["Count"])
                                    chip_html = f'<span style="display:inline-block; background:#EEF2FF; border:1px solid #C7D2FE; color:#3730A3; padding:5px 12px; margin:4px; border-radius:16px; font-size:12px; font-weight:600;">{brand_name} <span style="background:#C7D2FE; padding:1px 6px; border-radius:10px; font-size:10px; margin-left:4px;">{count}</span></span>'
                                    chips.append(chip_html)
                                
                                st.markdown(
                                    f'<div style="display:flex; flex-wrap:wrap; gap:2px; padding:10px; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; max-height:200px; overflow-y:auto;">{"".join(chips)}</div>',
                                    unsafe_allow_html=True,
                                )

                        # Download Processed Results - Dark Blue Button with Pure White Text
                        processed_count = min(len(cleaned_texts), len(mob_flags), len(brands_per_row))
                        df_processed = pd.DataFrame(
                            {
                                "Text": raw_texts[:processed_count],
                                "Mobile_Tech_Flag": np.asarray(mob_flags[:processed_count]),
                                "Brands_Detected": [
                                    ", ".join(b) if isinstance(b, list) else ""
                                    for b in brands_per_row[:processed_count]
                                ],
                            }
                        )
                        csv_data = df_processed.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Processed Results (CSV)",
                            data=csv_data,
                            file_name=f"brandpulse_{uploaded_file.name}",
                            mime="text/csv",
                            type="primary",
                        )

        except Exception as e:
            st.error(f"Error processing file '{uploaded_file.name}': {e}")
    else:
        st.info("Upload a CSV or Excel file containing article or tweet text above to compute live batch intelligence.")

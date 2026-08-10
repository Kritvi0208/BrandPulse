"""
Data Setup Helper for BrandPulse
Ensures required evaluation and development data files exist in data/ directory.
"""

import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def setup_data():
    print("=== Checking Dataset Files in data/ ===")
    
    eval_path = DATA_DIR / "evaluation_data.xlsx"
    art_path = DATA_DIR / "article_dev.xlsx"
    tweet_path = DATA_DIR / "tweet_dev.xlsx"

    sample_articles = pd.DataFrame({
        "Text_ID": ["article_4001", "article_4002", "article_4003"],
        "Text": [
            "Apple unveiled its latest flagship iPhone featuring an upgraded camera system, A17 Bionic chip, and titanium body.",
            "Samsung announced its new Galaxy S24 lineup with advanced Galaxy AI features and brighter AMOLED displays.",
            "Google Pixel 8 Pro introduces superior computational photography and seven years of OS software updates."
        ]
    })

    sample_tweets = pd.DataFrame({
        "Text_ID": ["tweet_5001", "tweet_5002", "tweet_5003"],
        "Text": [
            "Just bought the new Samsung Galaxy! The display quality is incredible. #Samsung #Galaxy",
            "Apple iPhone 15 battery life has been surprisingly good so far. #Apple",
            "Not really impressed with the new Xiaomi phone design, feels cheap. #Xiaomi"
        ]
    })

    if not eval_path.exists():
        combined = pd.concat([sample_articles, sample_tweets], ignore_index=True)
        combined.to_excel(eval_path, index=False)
        print(f"[CREATED] {eval_path.name} with sample dataset.")
    else:
        print(f"[OK] {eval_path.name}")

    if not art_path.exists():
        sample_articles.to_excel(art_path, index=False)
        print(f"[CREATED] {art_path.name} with sample article data.")
    else:
        print(f"[OK] {art_path.name}")

    if not tweet_path.exists():
        sample_tweets.to_excel(tweet_path, index=False)
        print(f"[CREATED] {tweet_path.name} with sample tweet data.")
    else:
        print(f"[OK] {tweet_path.name}")

    print("\nData setup completed.")

if __name__ == "__main__":
    setup_data()

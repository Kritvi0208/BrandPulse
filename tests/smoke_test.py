"""
Smoke Test for BrandPulse System
Verifies imports, preprocessing, brand detection, binary classification,
sentiment inference, and headline generation in a clean local environment.
"""

import os
import sys
import unittest
from pathlib import Path

# Add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class TestBrandPulsePipeline(unittest.TestCase):

    def test_01_imports(self):
        """Test core library and src module imports."""
        import spacy
        import transformers
        import torch
        import brands
        import utils
        import Article_Binary_Classifier_Inference
        import Tweet_Binary_Classifier_Inference
        import headline_generation
        import sentiment_inference

        self.assertTrue(True, "All imports succeeded.")

    def test_02_preprocessing(self):
        """Test text cleaning and language detection."""
        import utils

        sample_article = "Check out the new iPhone 15 at http://example.com! © Apple Inc."
        cleaned = utils.clean_articles([sample_article])
        self.assertNotIn("http", cleaned[0])
        self.assertNotIn("©", cleaned[0])

        sample_tweet = "Loving the new @Samsung Galaxy S24 Ultra! RT @tech_news http://t.co/123"
        cleaned_tweet = utils.clean_tweets([sample_tweet])
        self.assertNotIn("RT", cleaned_tweet[0])
        self.assertIn("Samsung", cleaned_tweet[0])

    def test_03_brand_detection(self):
        """Test brand entity recognition and indexing."""
        import brands

        text = "Apple and Samsung announced new flagship devices today, competing with Xiaomi."
        found_brands = brands.get_brands([text], verbose=False)[0]
        self.assertIn("apple", found_brands)
        self.assertIn("samsung", found_brands)
        self.assertIn("xiaomi", found_brands)

    def test_04_binary_classifiers(self):
        """Test mobile tech article and tweet binary classifiers."""
        import pandas as pd
        import Article_Binary_Classifier_Inference
        import Tweet_Binary_Classifier_Inference

        df_art = pd.DataFrame({
            "Text": ["Apple launched the new iPhone 15 with A17 chip.", "Stock market closed today."],
            "brands": [["apple"], []],
            "num_brands": [1, 0]
        })
        res_art = Article_Binary_Classifier_Inference.mobile_tech_binary_classifier(df_art)
        self.assertIn("Mobile_Tech", res_art.columns)

        df_tweet = pd.DataFrame({
            "Text": ["Loving my new Samsung phone!", "Good morning world!"],
            "brands": [["samsung"], []],
            "num_brands": [1, 0]
        })
        res_tweet = Tweet_Binary_Classifier_Inference.mobile_tech_binary_classifier(df_tweet)
        self.assertIn("Mobile_Tech", res_tweet.columns)

    def test_05_sentiment_inference(self):
        """Test sentiment classification pipeline."""
        import sentiment_inference

        classifier = sentiment_inference.SentimentClassifier(bert_path=None)
        tweet_data = [{"Text_ID": "t1", "Text": "Great performance on the new Apple M3 chip!"}]
        outs = classifier.predict(tweet_data, is_tweets=True)
        self.assertEqual(len(outs), 1)
        self.assertIn("Text_ID", outs[0])

    def test_06_headline_generation(self):
        """Test headline generator predict pipeline."""
        import headline_generation

        gen = headline_generation.headline_gen(device="cpu", path=None)
        articles = ["Apple announced the new iPhone 15 with A17 Bionic chip and upgraded cameras."]
        headlines = gen.predict(articles)
        self.assertEqual(len(headlines), 1)
        self.assertIsInstance(headlines[0], str)

if __name__ == "__main__":
    unittest.main()

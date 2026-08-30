"""
===============================================================================
Script: 01_ABSA_Extraction.py
Description: Extract daily overall sentiment polarity and aspect-based 
             negative sentiment counts (Hygiene, Service, Food, Ambience, Price)
             from Yelp restaurant reviews using RoBERTa.
===============================================================================
"""

import os
import json
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & HYPERPARAMETERS
# -----------------------------------------------------------------------------
DATASET_PATH = "yelp_academic_dataset_review.json"  # Path to Yelp Review JSON
OUTPUT_CSV_PATH = "daily_sentiment_features.csv"
SAMPLE_SIZE = 50000  # Number of reviews to process for quick execution (Set None for full)
BATCH_SIZE = 64
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"[*] Initializing ABSA Extraction pipeline on device: {'GPU' if DEVICE == 0 else 'CPU'}...")

# -----------------------------------------------------------------------------
# 2. ASPECT TAXONOMY & KEYWORDS (Pontiki et al., 2014 & SemEval Restaurant Taxonomy)
# -----------------------------------------------------------------------------
ASPECT_KEYWORDS = {
    'hygiene': [
        'dirty', 'clean', 'hygiene', 'bathroom', 'restroom', 'toilet', 'bug', 
        'cockroach', 'rat', 'hair', 'filthy', 'sanitary', 'smell', 'odor', 'gross', 'sick'
    ],
    'service': [
        'waiter', 'waitress', 'staff', 'service', 'manager', 'attentive', 'slow', 
        'rude', 'server', 'time', 'table', 'host', 'seated', 'orders', 'friendly'
    ],
    'food': [
        'food', 'taste', 'flavor', 'delicious', 'cold', 'salty', 'bland', 'overcooked', 
        'undercooked', 'dish', 'menu', 'meat', 'chicken', 'soup', 'fresh', 'portion'
    ],
    'ambience': [
        'ambience', 'atmosphere', 'noise', 'loud', 'decor', 'music', 'lighting', 
        'seating', 'cozy', 'cramped', 'vibe', 'environment'
    ],
    'price': [
        'price', 'expensive', 'cheap', 'overpriced', 'value', 'cost', 'bill', 
        'check', 'charged', 'money', 'worth'
    ]
}

# -----------------------------------------------------------------------------
# 3. LOAD DATASET (Yelp Open Dataset)
# -----------------------------------------------------------------------------
def load_yelp_reviews(file_path, sample_n=None):
    """Loads JSON review data and parses required fields."""
    if not os.path.exists(file_path):
        print(f"[!] Warning: '{file_path}' not found. Generating mock Yelp reviews for testing...")
        return generate_mock_yelp_data()

    print(f"[*] Reading reviews from {file_path}...")
    reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_n and i >= sample_n:
                break
            data = json.loads(line)
            reviews.append({
                'date': data['date'].split(' ')[0], # YYYY-MM-DD
                'text': data['text'],
                'stars': data['stars']
            })
    return pd.DataFrame(reviews)


def generate_mock_yelp_data():
    """Fallback generator to create mock review data if real file is missing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    sample_texts = [
        "The food was delicious, but the bathroom was dirty and disgusting!",
        "Very slow service, wait staff was super rude. Overpriced bill.",
        "Clean restaurant, friendly staff, and fresh meat. Highly recommended!",
        "Found a hair in my soup, terrible hygiene. Never coming back.",
        "Great atmosphere, nice music, and reasonable prices for dinner."
    ]
    data = []
    for d in dates:
        for _ in range(np.random.randint(3, 8)):
            data.append({
                'date': d.strftime('%Y-%m-%d'),
                'text': np.random.choice(sample_texts),
                'stars': np.random.randint(1, 6)
            })
    return pd.DataFrame(data)

df_reviews = load_yelp_reviews(DATASET_PATH, sample_n=SAMPLE_SIZE)
print(f"[+] Loaded {len(df_reviews)} total reviews across {df_reviews['date'].nunique()} unique dates.")

# -----------------------------------------------------------------------------
# 4. LOAD ROBERTA TRANSFORMER MODEL
# -----------------------------------------------------------------------------
# cardiffnlp/twitter-roberta-base-sentiment-latest
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"[*] Loading RoBERTa model '{MODEL_NAME}'...")

sentiment_task = pipeline(
    "sentiment-analysis", 
    model=MODEL_NAME, 
    tokenizer=MODEL_NAME, 
    top_k=None, 
    device=DEVICE,
    truncation=True,
    max_length=512
)

# Label conversion maps RoBERTa outputs into numerical polarity [-1, 0, 1]
LABEL_MAP = {
    'negative': -1.0,
    'neutral': 0.0,
    'positive': 1.0,
    'LABEL_0': -1.0,
    'LABEL_1': 0.0,
    'LABEL_2': 1.0
}

# -----------------------------------------------------------------------------
# 5. ASPECT-BASED SENTIMENT EXTRACTION LOGIC
# -----------------------------------------------------------------------------
def split_sentences(text):
    """Splits a review text into individual sentences for clause-level ABSA."""
    return [s.strip() for s in re.split(r'[.!?]+', text.lower()) if len(s.strip()) > 3]

def process_reviews_absa(df):
    """Applies sentence-level ABSA and extracts daily aggregates."""
    print("[*] Running ABSA pipeline on review sentences...")
    
    # Store sentence-level tuples: (date, sentence_text, aspect_matched)
    extracted_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Aspects"):
        date = row['date']
        sentences = split_sentences(row['text'])
        
        for sent in sentences:
            # Check which operational aspects are mentioned in sentence
            matched_aspects = []
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(kw in sent for kw in keywords):
                    matched_aspects.append(aspect)
            
            # Save record if sentence hits at least one aspect keyword
            for aspect in matched_aspects:
                extracted_records.append({
                    'date': date,
                    'sentence': sent,
                    'aspect': aspect
                })

    df_aspects = pd.DataFrame(extracted_records)
    print(f"[+] Found {len(df_aspects)} aspect-specific sentence occurrences.")

    if df_aspects.empty:
        raise ValueError("No aspect keywords matched in dataset. Check input texts.")

    # Batch prediction on aspect sentences via RoBERTa
    sentences_list = df_aspects['sentence'].tolist()
    batch_results = []
    
    print("[*] Inferring RoBERTa sentiment predictions in batches...")
    for i in tqdm(range(0, len(sentences_list), BATCH_SIZE), desc="Transformer Batches"):
        batch = sentences_list[i : i + BATCH_SIZE]
        preds = sentiment_task(batch)
        batch_results.extend(preds)

    # Process predicted labels and confidence scores
    processed_sentiments = []
    is_negative_flags = []

    for pred in batch_results:
        # Sort predictions by highest probability
        top_pred = sorted(pred, key=lambda x: x['score'], reverse=True)[0]
        label = top_pred['label'].lower()
        score = top_pred['score']
        
        polarity = LABEL_MAP.get(label, 0.0)
        processed_sentiments.append(polarity * score)  # Weighted Polarity
        is_negative_flags.append(1 if (polarity < 0 and score > 0.5) else 0)

    df_aspects['polarity'] = processed_sentiments
    df_aspects['is_negative'] = is_negative_flags

    # -------------------------------------------------------------------------
    # 6. DAILY AGGREGATION & FEATURE MATRIX GENERATION
    # -------------------------------------------------------------------------
    print("[*] Aggregating features into daily time-series dataset...")
    
    # Group overall polarity score by date
    daily_overall = df_aspects.groupby('date')['polarity'].mean().reset_index()
    daily_overall.rename(columns={'polarity': 'avg_polarity'}, inplace=True)

    # Pivot table for negative aspect counts per day
    daily_aspect_neg = df_aspects.pivot_table(
        index='date', 
        columns='aspect', 
        values='is_negative', 
        aggfunc='sum', 
        fill_value=0
    ).reset_index()

    # Rename columns to standard feature names used in Models 5, 6, and 7
    aspect_col_map = {
        'hygiene': 'neg_hygiene_count',
        'service': 'neg_service_count',
        'food': 'neg_food_count',
        'ambience': 'neg_ambience_count',
        'price': 'neg_price_count'
    }
    daily_aspect_neg.rename(columns=aspect_col_map, inplace=True)

    # Ensure all aspect columns exist even if zero occurrences
    for col in aspect_col_map.values():
        if col not in daily_aspect_neg.columns:
            daily_aspect_neg[col] = 0

    # Merge overall polarity with negative aspect counts
    daily_features = pd.merge(daily_overall, daily_aspect_neg, on='date', how='outer').fillna(0)
    daily_features.sort_values(by='date', inplace=True)

    return daily_features

# -----------------------------------------------------------------------------
# 7. EXECUTION AND SAVE OUTPUT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df_daily_absa = process_reviews_absa(df_reviews)
    
    # Export final dataset
    df_daily_absa.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n" + "="*80)
    print(f"[SUCCESS] Pipeline complete! File saved to: {OUTPUT_CSV_PATH}")
    print("="*80)
    print("\nPreview of Output Data (`daily_sentiment_features.csv`):")
    print(df_daily_absa.head(10))
    print("\nDataset Summary Stats:")
    print(df_daily_absa.describe())

"""
Open Google Colab and select a GPU runtime (Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU).

Install required transformer packages:
!pip install -q transformers datasets torch tqdm pandas numpy

Run the script:
!python 01_ABSA_Extraction.py

Output File
The script outputs daily_sentiment_features.csv, which contains the exact feature set required for Notebook 2 / Script 2 (02_Forecasting_and_SHAP.py):

date: YYYY-MM-DD
avg_polarity: Daily average sentiment polarity score
neg_hygiene_count: Total negative hygiene review mentions on that day
neg_service_count: Total negative service review mentions on that day
neg_food_count: Total negative food review mentions on that day
neg_ambience_count: Total negative ambience review mentions on that day
neg_price_count: Total negative price review mentions on that day
"""
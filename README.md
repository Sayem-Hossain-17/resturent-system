# Restaurant Demand Forecasting & Optimization System

An integrated AI framework for restaurant demand forecasting, aspect-based sentiment analysis (ABSA), and operational optimization, built as part of a thesis research project.

## Overview

This system combines natural language processing, time-series forecasting, and mathematical optimization to help restaurants make data-driven operational decisions. The pipeline ingests Yelp reviews, extracts aspect-level sentiment, forecasts future demand, and optimizes staffing and inventory decisions.

## Architecture

```
01_ABSA_Extraction.py         → Extracts daily sentiment features from Yelp reviews
02_Forecasting_and_SHAP.py    → Trains 7 forecasting models + SHAP explainability
03_Optimization_and_Dashboard.py → Staff scheduling, inventory optimization, Streamlit dashboard
daily_sentiment_features.csv  → Output ABSA features used by downstream modules
```

## Pipeline Stages

### 1. ABSA Extraction (`01_ABSA_Extraction.py`)

- Extracts daily overall sentiment polarity and aspect-based negative sentiment counts from Yelp reviews using a RoBERTa-based pipeline
- Aspect taxonomy: Hygiene, Service, Food, Ambience, Price
- Outputs `daily_sentiment_features.csv` with daily aggregated sentiment metrics

### 2. Forecasting & Explainability (`02_Forecasting_and_SHAP.py`)

- End-to-end training, evaluation, and explainability pipeline for 7 models:
  - ARIMA (traditional time-series)
  - Prophet (Facebook's forecasting tool)
  - XGBoost variants with lag features
  - LSTM (deep learning sequence model)
- Includes SHAP analysis for model interpretability
- Produces `model_performance_scorecard.csv`

### 3. Optimization & Dashboard (`03_Optimization_and_Dashboard.py`)

- **Staff Scheduling**: Linear programming (PuLP) to minimize labor costs while meeting forecasted demand across Morning, Evening, and Night shifts
- **Inventory Optimization**: Newsvendor model for optimal stock levels
- **Interactive Dashboard**: Streamlit web application for visualization and decision support

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install torch transformers pandas numpy matplotlib scipy statsmodels prophet xgboost scikit-learn shap pulp streamlit
```

## Usage

```bash
# Step 1: Extract sentiment features from Yelp reviews
python 01_ABSA_Extraction.py

# Step 2: Train forecasting models and generate SHAP explanations
python 02_Forecasting_and_SHAP.py

# Step 3: Launch optimization engine and dashboard
streamlit run 03_Optimization_and_Dashboard.py
```

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (optional, falls back to CPU)
- Yelp Academic Dataset (`yelp_academic_dataset_review.json`) for full extraction

## Project Structure

| File                               | Purpose                                             |
| ---------------------------------- | --------------------------------------------------- |
| `01_ABSA_Extraction.py`            | Aspect-based sentiment extraction from Yelp reviews |
| `02_Forecasting_and_SHAP.py`       | Demand forecasting with 7 ML models + SHAP          |
| `03_Optimization_and_Dashboard.py` | LP optimization and Streamlit dashboard             |
| `daily_sentiment_features.csv`     | Generated ABSA feature dataset                      |

## Author

Thesis research project — Restaurant Demand Forecasting & Optimization System

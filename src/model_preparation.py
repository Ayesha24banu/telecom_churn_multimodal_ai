# MODEL PREPARATION (Scaling + TF-IDF + Train/Test Split)
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def prepare_model_data(df):
    """
    Prepares data for Machine Learning & Deep Learning models.
     ML → Predicts CHURN  → (Structured data only)
     DL → Predicts SENTIMENT → (Text data only, NO churn info)
   
    Returns:
        X_train_ml, X_test_ml, y_train_ml, y_test_ml,
        X_train_dl, X_test_dl, y_train_dl, y_test_dl,
        scaler, tfidf
    """
    logging.info("Model preparation started...")

    try:
        # Copy dataset to prevent modification
        df_model = df.copy()
        logging.info("Dataset copied successfully.")

        
        # CREATE SENTIMENT LABEL → DL TARGET ONLY
        conditions = [
            (df_model["sentiment_compound"] >= 0.5),                  # strong positive
            (df_model["sentiment_compound"] > 0) & (df_model["sentiment_compound"] < 0.5),  # mid-positive
            (df_model["sentiment_compound"] > -0.5) & (df_model["sentiment_compound"] <= 0), # neutral
            (df_model["sentiment_compound"] <= -0.5)                  # strong negative 
        ]
        labels = [0, 1, 1, 2]  # 3-level sentiment (0=positive,1=neutral,2=negative)
        df_model["sentiment_label"] = np.select(conditions, labels)
        logging.info("Sentiment labels successfully assigned.")

        
        # TARGET COLUMNS (Keep Separate to Prevent Leakage)
        target_ml = "Churn"            # ML model predicts churn
        target_dl = "sentiment_label"  # DL model classifies Sentiment prediction
        text_col = "CustomerFeedback"  # for DL model

        # Numeric columns ONLY for ML model
        numeric_cols = [
            "tenure", "MonthlyCharges", "TotalCharges",
            "feedback_length", "word_count",
            "sentiment_pos", "sentiment_neg", "sentiment_neu",
            "sentiment_compound", "is_high_value_customer", "is_new_customer"
        ]

        # Split dataset into ML & DL inputs
        X_structured = df_model[numeric_cols]  # Structured for ML
        X_text = df_model[text_col]            # Text for DL
        y_ml = df_model[target_ml]             # Target for ML
        y_dl = df_model[target_dl]             # Target for DL
        logging.info("Data split into structured & text successfully.")

        
        # TRAIN–TEST SPLIT FOR ML 
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
            X_structured, y_ml, test_size=0.2, random_state=42, stratify=y_ml
        )

        #  TRAIN–TEST SPLIT FOR DL 
        X_train_dl_text, X_test_dl_text, y_train_dl, y_test_dl = train_test_split(
            X_text, y_dl, test_size=0.2, random_state=42, stratify=y_dl
        )

        logging.info(f"ML Train Shape: {X_train_ml.shape} | ML Test Shape: {X_test_ml.shape}")
        logging.info(f"DL Train Shape: {X_train_dl_text.shape} | DL Test Shape: {X_test_dl_text.shape}")

        
        # STANDARD SCALING → ONLY FOR ML FEATURES
        scaler = StandardScaler()
        X_train_ml_scaled = scaler.fit_transform(X_train_ml)
        X_test_ml_scaled = scaler.transform(X_test_ml)

        # Convert to sparse matrix
        X_train_ml = csr_matrix(X_train_ml_scaled)
        X_test_ml = csr_matrix(X_test_ml_scaled)
        logging.info("ML scaling completed successfully.")


        # TF-IDF FOR TEXT (DL) → Without Leakage 
        leak_words = ["cancel", "churn", "refund", "quit", "terminate", "leave"]
        tfidf = TfidfVectorizer(
            max_features=500,          # 🔥 balanced & general
            ngram_range=(1, 2),        # unigrams + bigrams
            min_df=3,                  # remove rare words
            stop_words=leak_words,     # REMOVE words that reveal churn!
        )
        X_train_dl = tfidf.fit_transform(X_train_dl_text)
        X_test_dl  = tfidf.transform(X_test_dl_text)
        logging.info("TF-IDF vectorization completed successfully.")

        # Return everything
        return X_train_ml, X_test_ml, X_train_dl, X_test_dl, y_train_ml, y_test_ml, y_train_dl, y_test_dl, scaler, tfidf

    except Exception as e:
        logging.error(f"Model Preparation Failed: {e}")
        print(" Model Preparation Failed. Check logs.")
        raise e

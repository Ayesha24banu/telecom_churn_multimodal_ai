# TELCO DATA FEATURE ENGINEERING
import pandas as pd
import numpy as np
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# MAIN FEATURE ENGINEERING FUNCTION
def feature_engineer_telco(df):
    """
    Creates:
    - Length-based features
    - VADER Sentiment features
    - Service count
    - Customer tags (new customer, high value)
    - Average monthly cost
    - Simple interaction features

    Returns:
        df_fe (DataFrame)
    """

    logging.info("Starting Feature Engineering...")
    
    try:
        df_fe = df.copy()

        # Feedback Length Based Features 
        df_fe["feedback_length"] = df_fe["CustomerFeedback"].astype(str).apply(len)
        df_fe["word_count"] = df_fe["CustomerFeedback"].astype(str).apply(lambda x: len(x.split()))

        logging.info("Text length features created.")

        # VADER Sentiment
        analyzer = SentimentIntensityAnalyzer()

        df_fe["sentiment_pos"] = df_fe["CustomerFeedback"].apply(lambda x: analyzer.polarity_scores(str(x))["pos"])
        df_fe["sentiment_neg"] = df_fe["CustomerFeedback"].apply(lambda x: analyzer.polarity_scores(str(x))["neg"])
        df_fe["sentiment_neu"] = df_fe["CustomerFeedback"].apply(lambda x: analyzer.polarity_scores(str(x))["neu"])
        df_fe["sentiment_compound"] = df_fe["CustomerFeedback"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

        # A simple sentiment flag for ML models
        df_fe["sentiment_flag"] = df_fe["sentiment_compound"].apply(
            lambda x: 1 if x > 0.05 else (-1 if x < -0.05 else 0)
        )

        logging.info("VADER sentiment features created.")

        #Service Count Feature
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

        available_services = [c for c in service_cols if c in df_fe.columns]
        df_fe["service_count"] = df_fe[available_services].sum(axis=1)

        logging.info("service_count feature created.")

        # Customer Value Features
        # Avg charges per month
        df_fe["avg_charges_per_month"] = df_fe.apply(
            lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else row["MonthlyCharges"],
            axis=1
        )

        # High-value customer (charges + tenure)
        df_fe["is_high_value_customer"] = ((df_fe["MonthlyCharges"] > 80) & (df_fe["tenure"] > 12)).astype(int)

        # New customer (tenure < 6 months)
        df_fe["is_new_customer"] = (df_fe["tenure"] < 6).astype(int)

        logging.info("Customer value features created.")

        # Simple Interaction Features
        df_fe["tenure_x_charges"] = df_fe["tenure"] * df_fe["MonthlyCharges"]
        df_fe["sentiment_x_charges"] = df_fe["sentiment_compound"] * df_fe["MonthlyCharges"]

        logging.info("Interaction features created.")

        logging.info("Feature Engineering COMPLETED successfully.")
        return df_fe

    except Exception as e:
        logging.error(f"Feature Engineering FAILED: {e}")
        print(" Feature Engineering Failed. Check logs.")
        raise e

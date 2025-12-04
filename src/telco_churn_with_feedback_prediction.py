# TELCO DATA PREDICTION
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

# Load model directories
from config import MODELS_DIR, OUTPUTS_DIR


logging.info(" telco_churn_with_feedback_prediction.py loaded.")

# LOAD ALL TRAINED MODELS & UTILITIES
print("\n Loading trained models...")
logging.info("Starting model loading...")

try:
    # File paths
    ml_model_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    dl_model_path = os.path.join(MODELS_DIR, "best_dl_model.h5")
    scaler_path   = os.path.join(MODELS_DIR, "scaler.pkl")
    tfidf_path    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    summary_path  = os.path.join(OUTPUTS_DIR, "models_evaluation", "model_summary.json")

    # Load models
    ml_model = joblib.load(ml_model_path)
    dl_model = tf.keras.models.load_model(dl_model_path)

    # Load pre-processing tools
    scaler = joblib.load(scaler_path)
    tfidf  = joblib.load(tfidf_path)

    # Load summary JSON (optional)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    print(" Models & utilities loaded successfully.")
    logging.info("Models & utilities loaded successfully.")

except Exception as e:
    logging.error(f" MODEL LOADING FAILED: {e}")
    raise RuntimeError("Model loading failed. Check file paths and trained models.")

    
# MAIN PREDICTION FUNCTION — MULTIMODAL
def predict_input(data):
    """
    Uses TWO MODELS:
      → ML MODEL       → Predicts CHURN (Structured Data)
      → DL MODEL (NLP) → Predicts SENTIMENT (Text Only)
      
    Unified Multimodal Prediction 
    If only feedback text → DL Sentiment Prediction
    If only structured data → ML Churn Prediction
    If BOTH → Return BOTH results
    """

    # Input TYPE validation
    if isinstance(data, dict):
        df = pd.DataFrame([data])
        logging.info("Input received as dict converted to DataFrame")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        logging.info("Input received as DataFrame")
    else:
        logging.error(" Invalid input format must be dict or DataFrame")
        raise ValueError("Please provide input as dict OR DataFrame.")

    result = {}  # final output dictionary


    # SENTIMENT ANALYSIS  → TEXT ONLY or FULL INPUT
    try:
        if "CustomerFeedback" in df.columns:
            feedback = df["CustomerFeedback"].astype(str)

            # TF-IDF transform → SAME AS TRAINING
            X_tf = tfidf.transform(feedback).toarray().reshape(-1, tfidf.max_features, 1)

            pred_prob = dl_model.predict(X_tf)
            pred_class = int(np.argmax(pred_prob, axis=1)[0]) # 0,1,2

            sentiment_map = {0: "Positive", 1: "Neutral", 2: "Negative"}
            result.update({
                "Sentiment Prediction": sentiment_map[pred_class],
                "Sentiment_Class": pred_class,
                "Model Used (DL)": "Deep Learning (Sentiment Analysis)"
            })
            logging.info("DL sentiment prediction successful.")
    except Exception as e:
        logging.error(f" Sentiment model prediction failed: {e}")
        result["sentiment_error"] = str(e)
    

    # CHURN PREDICTION → NUMERIC STRUCTURED DATA
    numeric_cols = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "feedback_length", "word_count",
        "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "sentiment_compound", "is_high_value_customer", "is_new_customer"
    ]

    try:
        if set(numeric_cols).issubset(df.columns):  # check ALL columns exist
            X_scaled = scaler.transform(df[numeric_cols])
            churn_prob = ml_model.predict_proba(X_scaled)[:, 1][0]
            churn_pred = int(churn_prob >= 0.5)

            result.update({
                "Churn Probability": round(float(churn_prob), 3),
                "Churn Prediction": "Yes - Will Churn" if churn_pred else "No - Safe Customer",
                "Model Used (ML)": "XGBoost (Churn Prediction)"
            })
            logging.info("ML churn prediction successful.")
        else:
            logging.warning("Numeric data missing Skipping churn prediction.")
    except Exception as e:
        logging.error(f" Churn model prediction failed: {e}")
        result["churn_error"] = str(e)

    #  NO VALID INPUT FOUND
    if not result:
        return {"error": "Input must have structured data OR 'CustomerFeedback' text."}

    return result
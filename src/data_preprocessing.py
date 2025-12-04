# TELCO DATA PREPROCESSING 
import pandas as pd
import numpy as np
import re
import logging
from sklearn.preprocessing import LabelEncoder

# TEXT CLEANING  HELPER FUNCTION
def clean_text(text):
    """
    Clean the CustomerFeedback text.
    - Lowercase
    - Remove punctuation
    - Remove extra spaces
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)     # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()        # clean extra spaces
    return text


# MAIN PREPROCESSING FUNCTION
def preprocess_telco(df):
    """
    Correct preprocessing for Telco + Customer Feedback NLP.

    Preprocessing Steps:
    1. Remove PromptInput if exists
    2. Clean numeric columns (tenure, MonthlyCharges, TotalCharges)
    3. Clean CustomerFeedback text
    4. Label encode categorical features (Except text)
    5. Encode target (Churn)

    Returns:
        processed_df, label_encoders
    """

    logging.info("Preprocessing Started...")

    try:
        df_p = df.copy()

        # Remove PromptInput (NOT needed)
        if "PromptInput" in df_p.columns:
            df_p = df_p.drop(columns=["PromptInput"])
            logging.info("PromptInput column removed.")
        else:
            logging.info("PromptInput column not found. Skipping removal.")

        # Clean numeric columns
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        for col in numeric_cols:
            if col in df_p.columns:
                df_p[col] = pd.to_numeric(df_p[col], errors="coerce")
                df_p[col] = df_p[col].fillna(df_p[col].median())
            else:
                logging.warning(f"Numeric column missing: {col}")

        logging.info("Numeric columns cleaned successfully.")

        # Text cleaning for CustomerFeedback
        if "CustomerFeedback" in df_p.columns:
            df_p["CustomerFeedback"] = df_p["CustomerFeedback"].apply(clean_text)
            logging.info("CustomerFeedback text cleaned.")
        else:
            logging.error("CustomerFeedback column missing!")
            raise KeyError("CustomerFeedback column missing in dataset.")

        # Label Encoding Categorical Columns 
        TARGET = "Churn"

        # All categorical columns except text + target
        categorical_cols = (
            df_p.select_dtypes(include="object")
            .columns.drop(["CustomerFeedback", TARGET], errors="ignore")
            .tolist()
        )
        
        # Encode Only proper feature categorical columnss
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df_p[col] = le.fit_transform(df_p[col].astype(str))
            label_encoders[col] = le

        logging.info("Categorical encoding completed.")

        # Encode Target Column (Churn)
        if TARGET in df_p.columns:
            le_target = LabelEncoder()
            df_p[TARGET] = le_target.fit_transform(df_p[TARGET].astype(str))
            label_encoders[TARGET] = le_target
            logging.info("Target column encoded successfully.")
        else:
            logging.error("Churn column missing!")
            raise KeyError("Target column 'Churn' missing.")
            
        return df_p, label_encoders

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        print(" Preprocessing failed. Check logs for details.")
        raise e
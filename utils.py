import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
import logging 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import  OUTPUTS_DIR, APP_LOG_PATH, LOGS_DIR


# Ensure log directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

#  LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,  # change to logging.DEBUG during debugging
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(APP_LOG_PATH, mode='a', encoding='utf-8'),  # 📝 Save to file
    ],
)
# Logger instance
logger = logging.getLogger(__name__)

# FINAL NUMERIC FEATURES FOR MODEL
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "feedback_length", "word_count",
    "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound",
    "is_high_value_customer", "is_new_customer",
]
logger.info("NUMERIC_COLS defined with %d features", len(NUMERIC_COLS))


# HELPER FUNCTIONS
analyzer = SentimentIntensityAnalyzer()   # VADER Analyzer

def _safe_text(x):
    """Utility: always return a clean string."""
    return "" if x is None else str(x)

def _vader_scores(text: str):
    """
    Compute VADER sentiment scores for a single text.
    Returns: pos, neg, neu, compound
    """
    scores = analyzer.polarity_scores(_safe_text(text))
    return scores["pos"], scores["neg"], scores["neu"], scores["compound"]

def prepare_single_input(data: dict, scaler):
    """
    Prepare user single input (adds VADER-derived features) and scale numeric columns.
    Returns: df, X_scaled, feedback_text
    """
    try:
        feedback = data.get("CustomerFeedback", "") or ""

        # VADER scores (same idea as feature_engineer_telco)
        pos, neg, neu, comp = _vader_scores(feedback)

        processed = {
            "tenure": float(data.get("tenure", 0)),
            "MonthlyCharges": float(data.get("MonthlyCharges", 0)),
            "TotalCharges": float(data.get("TotalCharges", 0)),
            "is_high_value_customer": 1 if data.get("is_high_value_customer") == "High Value" else 0,
            "is_new_customer": 1 if data.get("is_new_customer") == "New" else 0,
            # original feedback
            "CustomerFeedback": feedback,

            #  Derived features – must match training
            "feedback_length": len(feedback),
            "word_count": len(feedback.split()) if feedback.strip() else 0,
            "sentiment_pos": pos,
            "sentiment_neg": neg,
            "sentiment_neu": neu,
            "sentiment_compound": comp,
        }

        # DataFrame with all required numeric features
        df = pd.DataFrame([processed])

        # Convert to numeric & scale
        df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)
        X_scaled = scaler.transform(df[NUMERIC_COLS])

        logger.info(" Single input processed successfully (with VADER features)")
        return df, X_scaled, feedback

    except Exception as e:
        logger.exception(" Error in prepare_single_input:")
        st.error(f"Error preparing input for prediction: {e}")
        return None, None, None

def run_single_prediction(data: dict, ml_model, dl_model, tfidf, scaler):
    """
     Run FULL prediction for a single user:
    1) Build engineered features (VADER + numeric) → ML churn
    2) Run DL sentiment model (TF-IDF + CNN-BiLSTM)
    3) Optional hybrid rule (if you want later)
       
    Returns: result dict, processed df, X_scaled
    """
    df, X_scaled, feedback = prepare_single_input(data, scaler)
    if df is None:
        st.error("❌ Prediction failed — invalid input.")
        return None, None, None
    
    try:
        # ML Churn Prediction
        churn_prob = float(ml_model.predict_proba(X_scaled)[0, 1])
        churn_label = "Yes – Likely to Churn" if churn_prob >= 0.5 else "No – Safe Customer"

        # Base result
        result = {"churn_label": churn_label, "churn_probability": round(churn_prob, 3)}

        # DL Sentiment Prediction (if feedback exists)
        if feedback.strip():
            X_tf = tfidf.transform([feedback])
            # Match training shape: (batch, timesteps, 1)
            arr = X_tf.toarray().reshape(-1, X_tf.shape[1], 1)

            probs = dl_model.predict(arr)
            pred_idx = int(np.argmax(probs, axis=1)[0])
            sentiment_map ={0:'Neutral', 1:'Positive',2:'Negative'}

            result["sentiment_label"] = sentiment_map[pred_idx]
        else:
            result["sentiment_label"] = "No Feedback Provided"

        logger.info("Single prediction completed successfully: %s", result)
        return result, df, X_scaled

    except Exception as e:
        logger.exception("Error in run_single_prediction:")
        st.error(f"Prediction failed: {e}")
        return None, None, None


def prepare_batch_df_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess batch CSV for ML & DL:
    - Ensures 'CustomerFeedback' exists
    - Computes VADER sentiment scores
    - Adds feedback_length, word_count
    - Ensures numeric types and NUMERIC_COLS presence
    """
    try:
        df = df.copy()

        # Handle feedback column safely
        if "CustomerFeedback" not in df.columns:
            df["CustomerFeedback"] = ""
        df["CustomerFeedback"] = df["CustomerFeedback"].fillna("").astype(str)

        # Text length features
        df["feedback_length"] = df["CustomerFeedback"].apply(len)
        df["word_count"] = df["CustomerFeedback"].apply(lambda x: len(x.split()) if x.strip() else 0)

        # VADER sentiment features (row-wise)
        pos_list, neg_list, neu_list, comp_list = [], [], [], []
        for text in df["CustomerFeedback"]:
            p, n, ne, c = _vader_scores(text)
            pos_list.append(p)
            neg_list.append(n)
            neu_list.append(ne)
            comp_list.append(c)

        df["sentiment_pos"] = pos_list
        df["sentiment_neg"] = neg_list
        df["sentiment_neu"] = neu_list
        df["sentiment_compound"] = comp_list

        # Flags → convert bool/nan to 0/1
        df["is_high_value_customer"] = df["is_high_value_customer"].map({"High Value": 1, "Regular": 0}).fillna(0)
        df["is_new_customer"] = df["is_new_customer"].map({"New": 1, "Existing": 0}).fillna(0)

        # Ensure numeric types
        for col in ["tenure", "MonthlyCharges", "TotalCharges",
                    "is_high_value_customer", "is_new_customer"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

        # Guarantee all NUMERIC_COLS exist
        for col in NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)

        logger.info("Batch data prepared successfully (VADER + numeric features)")
        return df
    
    except Exception as e:
        logger.exception("Error in prepare_batch_df_for_model:")
        st.error(f"Batch preparation failed: {e}")
        return pd.DataFrame()  # return EMPTY — prevents crash


def run_batch_predictions(df_input: pd.DataFrame, scaler, ml_model, dl_model, tfidf):
    """
     Run predictions for an entire dataset (Batch):

    1) Prepare batch data (VADER + numeric feature engineering)
    2) ML churn prediction for all rows
    3) DL sentiment prediction (if CustomerFeedback exists)
    """
    try:
        df_prepared = prepare_batch_df_for_model(df_input)

        if df_prepared.empty:
            st.error("⚠ Batch preparation failed — no valid rows.")
            return pd.DataFrame()

        # Sanity check
        missing = [c for c in NUMERIC_COLS if c not in df_prepared.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        # ML churn for all rows
        X_scaled = scaler.transform(df_prepared[NUMERIC_COLS])
        churn_probs = ml_model.predict_proba(X_scaled)[:, 1]

        df_prepared["Churn_Probability"] = np.round(churn_probs, 3)
        df_prepared["Churn_Label"] = np.where(churn_probs >= 0.5, "Yes – Likely to Churn", "No – Safe Customer")

        # DL sentiment for all rows (if feedback present)
        sentiments = []
        if "CustomerFeedback" in df_prepared.columns:
            for text in df_prepared["CustomerFeedback"].fillna(""):
                if text.strip():
                    X_tf = tfidf.transform([text])
                    arr = X_tf.toarray().reshape(-1, X_tf.shape[1], 1)
                    probs = dl_model.predict(arr)
                    sentiment_map = {0:'Neutral', 1:'Positive',2:'Negative'}

                    sentiments.append(sentiment_map[int(np.argmax(probs))])
                else:
                    sentiments.append("No Feedback")
        else:
            sentiments = ["No Feedback"] * len(df_prepared)

        df_prepared["Sentiment_Label"] = sentiments

        logger.info("Batch predictions completed – rows: %s", len(df_prepared))
        return df_prepared
    
    except Exception as e:
        logger.exception("Error in run_batch_predictions:")
        st.error(f"Batch prediction failed: {e}")
        return pd.DataFrame()  # return EMPTY


def add_single_history(input_data, result, conn, cursor):
    """
     Save prediction to:
    1) Session history
    2) Permanent SQLite DB
    """
    try:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "single",
            "customer_id": input_data.get("customerID") or input_data.get("customer_id"),
            "gender": input_data.get("gender"),
            "SeniorCitizen": input_data.get("SeniorCitizen"),
            "Partner": input_data.get("Partner"),
            "Dependents": input_data.get("Dependents"),
            "Contract": input_data.get("Contract"),
            "InternetService": input_data.get("InternetService"),
            "PaymentMethod": input_data.get("PaymentMethod"),
            "PaperlessBilling": input_data.get("PaperlessBilling"),
            "tenure": input_data.get("tenure"),
            "MonthlyCharges": input_data.get("MonthlyCharges"),
            "TotalCharges": input_data.get("TotalCharges"),
            "is_high_value_customer": input_data.get("is_high_value_customer"),
            "is_new_customer": input_data.get("is_new_customer"),
            "churn_label": result.get("churn_label"),
            "churn_probability": result.get("churn_probability"),
            "sentiment_label": result.get("sentiment_label"),
        }

        # Save in session
        st.session_state["history"]["single"].append(record)

        # Save in SQLite DB
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, mode, customer_id, gender, SeniorCitizen, Partner, Dependents, Contract, 
            InternetService, PaymentMethod, PaperlessBilling, tenure, MonthlyCharges, TotalCharges, 
            is_high_value_customer, is_new_customer, churn_label, churn_probability, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"], record["mode"], record["customer_id"], record["gender"], record["SeniorCitizen"],
            record["Partner"], record["Dependents"], record["Contract"], record["InternetService"],
            record["PaymentMethod"], record["PaperlessBilling"],
            record["tenure"], record["MonthlyCharges"], record["TotalCharges"],
            record["is_high_value_customer"], record["is_new_customer"],
            record["churn_label"], record["churn_probability"], record["sentiment_label"],
        ))
        conn.commit()

        logger.info("Single history added to DB")
    except Exception as e:
        logger.exception("Error saving single history:")
        st.error(f"History save failed: {e}")


def add_batch_history(df_scored: pd.DataFrame, cursor, conn, filename: str = "uploaded.csv"):
    """
    Log batch prediction results to:
    1) Session State
    2) SQLite Database (each row)
    """
    try:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "batch",
            "file_name": filename,
            "rows": len(df_scored),
            "churn_rate": float(df_scored["Churn_Probability"].mean()) * 100
            if "Churn_Probability" in df_scored.columns else None,
            "with_feedback": "CustomerFeedback" in df_scored.columns,
        }

        st.session_state["history"]["batch"].append(record)

        # Save individual rows
        for _, row in df_scored.iterrows():
            # --- FIX: CONVERT NUMERIC FLAGS → TEXT BEFORE SAVING ---
            hv_text = "High Value" if row.get("is_high_value_customer", 0) == 1 else "Regular"
            nc_text = "New" if row.get("is_new_customer", 0) == 1 else "Existing"

            cursor.execute("""
                    INSERT INTO predictions (
                    timestamp, mode, customer_id, gender, SeniorCitizen, Partner, Dependents,
                    Contract, InternetService, PaymentMethod, PaperlessBilling,
                    tenure, MonthlyCharges, TotalCharges, is_high_value_customer, is_new_customer,
                    churn_label, churn_probability, sentiment_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["timestamp"], record["mode"], row.get("customerID") or row.get("customer_id"), row.get("gender"),
                row.get("SeniorCitizen"), row.get("Partner"), row.get("Dependents"), row.get("Contract"), row.get("InternetService"),
                row.get("PaymentMethod"), row.get("PaperlessBilling"), row.get("tenure"), row.get("MonthlyCharges"), row.get("TotalCharges"),
                hv_text, nc_text, row.get("Churn_Label"), row.get("Churn_Probability"),
                row.get("Sentiment_Label", "No Feedback")
            ))
            conn.commit()

            logger.info("Batch history saved to DB successfully")

    except Exception as e:
        logger.exception("Error saving batch history:")
        st.error(f"Batch history save failed: {e}")


def backup_history_csv(conn):
    """
     Automatically backup SQLITE DB to CSV — daily snapshots.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        backup_file = os.path.join(OUTPUTS_DIR, f"history_backup_{today}.csv")

        df_db = pd.read_sql("SELECT * FROM predictions", conn)
        df_db.to_csv(backup_file, index=False)

        st.success(f"📦 Backup saved: {backup_file}")
        logger.info("Backup completed")

    except Exception as e:
        logger.exception("Backup failed:")
        st.error(f"Backup failed: {e}")


# 🔍 AUTO FEATURE INFLUENCE INSIGHTS 
def generate_feature_insights(shap_vals, feature_names):
    """
    Auto-generate insights from SHAP values
    - Computes feature mean impact
    - Highlights top churn influencers
    - Used in GLOBAL SHAP TAB
    """
    try:
        shap_mean = np.mean(np.abs(shap_vals), axis=0)
        df_insights = pd.DataFrame({"Feature": feature_names, "Mean_SHAP_Impact": shap_mean}).sort_values("Mean_SHAP_Impact", ascending=False)

        st.markdown("### 📌 **Top Drivers of Churn**")
        st.dataframe(df_insights)  # cleaner display

        # Only Top 3 Highlights
        st.markdown("#### 🔥 Key Insights (Auto-Generated):")
        for _, row in df_insights.head(3).iterrows():
            st.write(f"👉 **{row['Feature']}** strongly influences churn — consider monitoring this feature.")
        
        logger.info("SHAP feature insights generated")
        return df_insights

    except Exception as e:
        logger.exception("Error in generate_feature_insights:")
        st.warning(f"⚠ Could not generate SHAP insights: {e}")
        return None


# SHAP SUPPORT FUNCTION
def _select_positive_class_shap(shap_vals):
    """
    SHAP returns different formats depending on explainer type.
    This function standardizes output → always returns `n_samples x n_features`
    Used in all SHAP functions.
    """
    try:
        if isinstance(shap_vals, list):
            return shap_vals[0]  # KernelExplainer returns [n_samples, n_features]
        if isinstance(shap_vals, np.ndarray):
            return shap_vals
        st.warning("⚠ Unknown SHAP format → Skipped.")
        return None
 
    except Exception as e:
        logger.exception("Error in _select_positive_class_shap:")
        return None


# GLOBAL SHAP — ACROSS DATASET
def plot_global_shap(explainer, shap_values_bg, bg_scaled_df):
    """Global SHAP summary – Overall feature importance across the dataset."""
    if explainer is None or shap_values_bg is None or bg_scaled_df is None:
        st.warning("⚠ SHAP is not available in this environment.")
        return

    st.subheader("🌍 Global SHAP Importance (XGBoost Model Only)")
    st.write("These features have the **highest mean influence** on churn across dataset.")

    shap_vals = _select_positive_class_shap(shap_values_bg)
    if shap_vals is None:
        st.error("❌ No SHAP values found — model may not support explainability.")
        return

    # GLOBAL SHAP BAR PLOT
    try:
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot( shap_vals, bg_scaled_df, feature_names=NUMERIC_COLS, plot_type="bar", show=False, max_display=10)
        st.pyplot(fig)                           
        plt.close(fig) 
        plt.close("all")
    
    except Exception as e:
        st.warning(f"⚠ Plot error: {e}")

    # TABLE FORMAT (More readable)
    mean_shap = np.mean(shap_vals, axis=0)
    contrib_table = pd.DataFrame({
        "Feature": NUMERIC_COLS,
        "Avg_SHAP_Impact": mean_shap
    }).sort_values("Avg_SHAP_Impact", key=lambda x: abs(x), ascending=False)

    st.markdown("### 📋 Global Feature Contribution Table")
    st.dataframe(contrib_table.reset_index(drop=True))  

    st.info("These features have the highest overall impact on churn across the dataset.")
    
    st.markdown("### 🤖 AI Observation:")
    top_feature = contrib_table.iloc[0]["Feature"]
    st.write(f"📌 **Most influential feature:** `{top_feature}`")

    logger.info("GLOBAL SHAP analysis shown successfully")

# SINGLE PREDICTION SHAP
def plot_single_shap_from_input(input_data: dict, explainer, scaler):
    """Compute & display SHAP explanation for 1 customer (last prediction)."""
    if explainer is None:
        st.warning("⚠ SHAP explainer is not available.")
        return

    df_single, X_scaled, _ = prepare_single_input(input_data, scaler)

    try:
        shap_vals_raw = explainer.shap_values(X_scaled)
    except Exception as e:
        logger.exception("SHAP error — single prediction:")
        st.error(f"Error computing SHAP values: {e}")
        return

    shap_vals = _select_positive_class_shap(shap_vals_raw)
    if shap_vals is None: 
        st.error("⚠ Invalid SHAP output — cannot parse values.")
        return

    # shap_vals should be (1, n_features)
    shap_arr = np.array(shap_vals).reshape(shap_vals.shape[0], shap_vals.shape[-1])
    shap_row = shap_arr[0].flatten()

    # TABLE — LOCAL FEATURE CONTRIBUTION
    contrib_df = pd.DataFrame({
        "Feature": NUMERIC_COLS,
        "Input_Value": df_single[NUMERIC_COLS].iloc[0].values,
        "SHAP_Contribution": shap_row,
        }
    ).sort_values("SHAP_Contribution", key=lambda x: abs(x), ascending=False)

    # Plot
    st.subheader("🧬 Local SHAP – Single Prediction")
    
    # BAR PLOT
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(contrib_df["Feature"], contrib_df["SHAP_Contribution"])
    ax.set_xlabel("SHAP Value (Impact on Churn Probability)")
    ax.set_ylabel("Feature")
    ax.set_title("Top Feature Impact on Churn (Single Prediction)")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)
    plt.close("all")

    # SHOW TABLE
    st.markdown("### 📋 Feature Contribution Table")
    st.dataframe(contrib_df.reset_index(drop=True))
    st.info("Higher SHAP impact means stronger influence on churn decision for this customer.")

    st.markdown("### 🤖 AI Insight:")
    top_feature = contrib_df.iloc[0]["Feature"]
    st.write(f"📌 **Most influential feature:** `{top_feature}` — this has the strongest effect on churn.")

    logger.info("SHAP single prediction displayed successfully")


# BATCH-LEVEL SHAP
def plot_batch_shap_from_df(df_batch: pd.DataFrame, explainer, scaler):
    """Run SHAP on multiple records — requires batch prediction first."""
    if explainer is None:
        st.warning("⚠ SHAP explainer is not available.")
        return
    
    if df_batch is None or df_batch.empty:
        st.info("No batch data available in this session.")
        return

    try:
        X_scaled = scaler.transform(df_batch[NUMERIC_COLS])
        shap_vals_raw = explainer.shap_values(X_scaled)
        shap_vals = _select_positive_class_shap(shap_vals_raw)
    except Exception as e:
        logger.exception("Error computing batch SHAP:")
        st.error(f"Failed to compute SHAP values: {e}")
        return

    st.subheader("📂 Batch SHAP Summary — Top Churn Influencers")
    
    # GLOBAL SHAP MEAN IMPACT TABLE
    mean_vals = np.mean(shap_vals, axis=0)
    contrib_df = pd.DataFrame({
        "Feature": NUMERIC_COLS,
        "Mean_Input_Value": df_batch[NUMERIC_COLS].mean(),
        "Mean_SHAP_Impact": mean_vals,
    }).sort_values("Mean_SHAP_Impact", key=lambda x: abs(x), ascending=False)

    # SHAP BAR PLOT
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.summary_plot(
            shap_vals,
            X_scaled,
            feature_names=NUMERIC_COLS,
            plot_type="bar",
            show=False,
            max_display=10
        )
        st.pyplot(fig)
        plt.close(fig); plt.close("all")
    except Exception as e:
        st.warning(f"⚠ Plot could not be rendered: {e}")

    st.markdown("### 📋 Batch Feature Contribution Table")
    st.dataframe(contrib_df.reset_index(drop=True))
    st.info("📌 Insights: High SHAP impact = Strong influence on churn across customers in this batch.")
    
    st.markdown("### 🤖 AI Insight:")
    top_feature = contrib_df.iloc[0]["Feature"]
    st.write(f"📌 **Most influential batch feature:** `{top_feature}` — this has the strongest effect on churn.")

    logger.info("Batch SHAP insights rendered.")


# CHURN vs NON-CHURN COMPARISON (Batch Only)    
def plot_churn_vs_nonchurn_shap(df_batch: pd.DataFrame, explainer, scaler):
    """
    Compare SHAP impact BETWEEN churned vs non-churned customers.
    Only shown in BATCH SHAP TAB — NOT suitable for Global or Single TAB.
    """
    if explainer is None:
        st.warning("⚠ SHAP explainer is not available.")
        return

    if df_batch is None or df_batch.empty:
        st.info("⚠ No batch data available in this session.")
        return

    if "Churn_Label" not in df_batch.columns:
        st.warning("⚠ 'Churn_Label' column missing — run batch prediction first.")
        return

    st.subheader("🆚 SHAP Comparison – Churned vs Non-Churned Customers (Batch)")

    # Only take numeric model features
    df_temp = df_batch.copy()
    X_scaled = scaler.transform(df_temp[NUMERIC_COLS])

    try:
        shap_vals_raw = explainer.shap_values(X_scaled)
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        return

    shap_vals = _select_positive_class_shap(shap_vals_raw)
    if shap_vals is None or shap_vals.shape[0] != df_temp.shape[0]:
        st.warning("⚠ Shape mismatch – SHAP could not align with dataset.")
        return

    # TOTAL SHAP IMPACT PER ROW
    df_temp["SHAP_Sum"] = np.abs(shap_vals).sum(axis=1)

    # BOXPLOT – CHURN vs NON-CHURN
    fig, ax = plt.subplots(figsize=(7, 4))
    df_temp.boxplot(column="SHAP_Sum", by="Churn_Label", ax=ax)
    ax.set_title("SHAP Impact – Churned vs Non-Churned Customers")
    ax.set_ylabel("Total SHAP Impact (Feature Influence)")
    plt.suptitle("")
    st.pyplot(fig)
    plt.close(fig); plt.close("all")

    # BUSINESS INSIGHTS (Auto Generated)
    churned_mean = df_temp[df_temp["Churn_Label"] == "Yes – Likely to Churn"]["SHAP_Sum"].mean()
    nonchurn_mean = df_temp[df_temp["Churn_Label"] == "No – Safe Customer"]["SHAP_Sum"].mean()

    st.markdown("---")
    st.markdown("### 🤖 AI Insight:")
    if churned_mean > nonchurn_mean:
        st.write(f"🔥 Churned customers show **higher decision instability (avg SHAP impact = {churned_mean:.3f})**,")
        st.write("🔎 This means their churn prediction is influenced by **multiple strong factors.**")
    else:
        st.write("✔ Non-churned customers show **less SHAP impact**, meaning the model is more confident.")

    st.info("Higher SHAP impact = stronger influence of multiple features on the churn decision.")
    logger.info("CHURN vs NON-CHURN analysis completed.")


def generate_business_report(summary_df,  company_name="TelcoWave Solutions Pvt Ltd"):
    """
    Generate a professional PDF report from Business Dashboard insights.
    
    Key Features:
    - Auto-generated business summary & recommendations
    - Clean PDF layout using ReportLab
    - Works with ANY summary_df (metric/value format)
    - Includes date & time for tracking
    - Fully reusable in real-world production apps

    Requirements:
    - summary_df must contain two columns → ["Metric", "Value"]
    - OUTPUTS_DIR must exist before running this function
    """
    try:
        # Ensure valid dataframe
        if summary_df is None or summary_df.empty:
            raise ValueError("summary_df is empty — cannot generate report.")

        # File Output Path
        file_path = os.path.join(OUTPUTS_DIR, "business_summary_report.pdf")

        # Setup PDF Layout
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()  # Predefined ReportLab styles
        elements = []

        # HEADER SECTION
        elements.append(Paragraph(f"<b>{company_name}</b>", styles["Title"]))
        elements.append(Paragraph("<b>📊 Telco AI – Business Insights Summary Report</b>", styles["Heading2"]))
        elements.append(Paragraph("Generated by Telco AI – Churn & Sentiment Intelligence System", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # EXECUTIVE SUMMARY
        exec_text = (
            "This report provides data-driven insights about customer behavior and churn risk. "
            "Using predictive Machine Learning and Sentiment Analysis, "
            "we identify retention opportunities and potential revenue loss. "
            "These insights can be used by CRM teams and Business leaders for real action."
        )
        elements.append(Paragraph("<b>🧠 Executive Summary</b>", styles["Heading3"]))
        elements.append(Paragraph(exec_text, styles["Normal"]))
        elements.append(Spacer(1, 18))

        # METRICS TABLE
        elements.append(Paragraph("<b>📌 Key Business Metrics</b>", styles["Heading3"]))

        # Ensure correct table format
        if list(summary_df.columns) != ["Metric", "Value"]:
            summary_df.columns = ["Metric", "Value"]

        data = [["Metric", "Value"]] + summary_df.values.tolist()
        elements.append(Table(data))
        elements.append(Spacer(1, 18))

        # RULE-BASED BUSINESS RECOMMENDATIONS
        elements.append(Paragraph("<b>🧠 Recommended Actions:</b>", styles["Heading3"]))
        recs = [
            "✔ High churn customers → Offer retention discounts / call follow-ups.",
            "✔ VIP customers → Introduce loyalty rewards & premium support.",
            "✔ Negative feedback → Escalate to customer support team.",
            "✔ Long tenure customers → Upsell higher service packages.",
        ]
        for r in recs:
            elements.append(Paragraph(r, styles["Normal"]))

        # TIMESTAMP
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(
            f"<b>⏱ Report Generated On:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        ))

        # BUILD PDF
        doc.build(elements)

        logger.info(f"Business report successfully created → {file_path}")
        return file_path

    except Exception as e:
        logger.exception("Error while generating business report:")
        st.error(f"🚨 Report generation failed: {e}")
        return None
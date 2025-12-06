import os
import logging
import sqlite3

import numpy as np
import pandas as pd
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st

from config import MODELS_DIR, DATA_FEATURED, OUTPUTS_DIR, APP_LOG_PATH, LOGS_DIR

from utils import (run_single_prediction, run_batch_predictions, generate_business_report, plot_global_shap, add_single_history, add_batch_history,
    backup_history_csv, generate_feature_insights, _select_positive_class_shap, plot_single_shap_from_input, plot_batch_shap_from_df, plot_churn_vs_nonchurn_shap,
)


# Ensure log directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,  # change to logging.DEBUG during debugging
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(APP_LOG_PATH, mode='a', encoding='utf-8'),  # Save to file
    ],
)
# Logger instance
logger = logging.getLogger(__name__)

logger.info("Starting Telco Customer AI – Streamlit app initialization")


# FILESYSTEM & DATABASE SETUP
# Ensure outputs directory exists for reports, backups, DB, etc.
try:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger.info("OUTPUTS_DIR ensured at: %s", OUTPUTS_DIR)
except Exception as e:
    logger.exception("Failed to create OUTPUTS_DIR at %s", OUTPUTS_DIR)
    st.error("Critical error: Unable to create outputs directory. Please check file permissions.")
    st.stop()

# SQLite DB path for persistent prediction history
DB_PATH = os.path.join(OUTPUTS_DIR, "prediction_history.db")

try:
    # check_same_thread=False ⇒ allow usage across Streamlit callbacks
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    logger.info("Connected to SQLite database at: %s", DB_PATH)

    # Create predictions table if it does not exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            customer_id TEXT,
            gender TEXT,
            SeniorCitizen TEXT,
            Partner TEXT,
            Dependents TEXT,
            Contract TEXT,
            InternetService TEXT,
            PaymentMethod TEXT,
            PaperlessBilling TEXT,
            tenure REAL,
            MonthlyCharges REAL,
            TotalCharges REAL,
            is_high_value_customer TEXT,
            is_new_customer TEXT,
            churn_label TEXT,
            churn_probability REAL,
            sentiment_label TEXT
        )
        """
    )
    conn.commit()
    logger.info("Verified/created 'predictions' table in SQLite DB")

except sqlite3.Error as e:
    logger.exception("Database initialization failed at %s", DB_PATH)
    st.error("Critical error: Unable to initialize history database. Please contact support.")
    st.stop()


# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="Telco AI – Churn & Sentiment",
    page_icon="📡",
    layout="wide",
)
logger.info("Streamlit page configured (wide layout, title, icon set)")

#  CONSTANTS – FEATURE COLUMNS (ML INPUTS): These are the exact numeric features expected by the ML churn model.
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "feedback_length", "word_count",
    "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound",
    "is_high_value_customer", "is_new_customer",
]
logger.info("NUMERIC_COLS defined with %d features", len(NUMERIC_COLS))

#  SESSION STATE – IN-MEMORY USER SESSION DATA
# history: keeps track of single & batch predictions in current Streamlit session
if "history" not in st.session_state:
    st.session_state["history"] = {"single": [], "batch": []}
    logger.info("Initialized session_state['history']")

# last_single_input: used for SHAP explainability (single prediction tab)
if "last_single_input" not in st.session_state:
    st.session_state["last_single_input"] = None
    logger.info("Initialized session_state['last_single_input']")

# last_batch_df: used for SHAP + batch analytics
if "last_batch_df" not in st.session_state:
    st.session_state["last_batch_df"] = None
    logger.info("Initialized session_state['last_batch_df']")

# LOAD MODELS & DATA (with Caching)
@st.cache_resource(show_spinner=True)
def load_assets():
    """
    Load ML/DL models, preprocessors, and the featured dataset.
    This is cached for performance & loaded only once per session.

    Returns:
        ml_model      → Trained ML churn model
        dl_model      → Trained DL sentiment model
        scaler        → Feature scaler used during model training
        tfidf         → Text vectorizer for sentiment model
        df_featured   → Dataset for dashboards / SHAP global
        explainer     → SHAP explainer (KernelExplainer)
        bg_scaled_df  → Scaled background dataset for SHAP
        shap_values_bg→ Precomputed SHAP values
    """
    logger.info("Loading ML/DL models and assets...")
    try:
        if not os.path.exists(MODELS_DIR):
            st.error("Model directory missing. Please check deployment files.")
            st.stop()

        # Load Models & Preprocessors
        ml_model = joblib.load(os.path.join(MODELS_DIR, "best_ml_model.pkl"))
        dl_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_dl_model.h5"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

        logger.info("Models & preprocessors loaded successfully")

    except Exception as e:
        logger.exception("Error loading models:")
        st.error("Critical Error: ML/DL model files missing or corrupted. Please verify model directory.")
        st.stop()

    # Load Featured Dataset
    try:
        df_featured = pd.read_csv(os.path.join(DATA_FEATURED, "featured_telco.csv"))
        logger.info("Featured dataset loaded: %s rows", len(df_featured))

        # Try to add churn predictions automatically 
        df_temp = df_featured.copy()
        df_temp[NUMERIC_COLS] = df_temp[NUMERIC_COLS].astype(float)
        X_scaled_temp = scaler.transform(df_temp[NUMERIC_COLS])
        churn_probs_temp = ml_model.predict_proba(X_scaled_temp)[:, 1]

        # Add to dataset
        df_featured["Churn_Probability"] = np.round(churn_probs_temp, 3)
        df_featured["Churn_Label"] = np.where(churn_probs_temp >= 0.5, "Yes – Likely to Churn", "No – Safe Customer")

    except Exception as e:
        logger.warning("Could not auto-generate churn predictions for featured dataset: %s", e)
        st.warning("⚠ Dataset loaded, but churn predictions could not be auto-generated.")

    # SHAP GLOBAL EXPLAINER
    explainer, bg_scaled_df, shap_values_bg = None, None, None
    try:
        df_ml = df_featured[NUMERIC_COLS].copy()

        # Sample 50 rows to use as SHAP background data
        n_sample = min(50, len(df_ml))
        bg_raw = df_ml.sample(n_sample, random_state=42)
        bg_scaled = scaler.transform(bg_raw)
        bg_scaled_df = pd.DataFrame(bg_scaled, columns=NUMERIC_COLS)

        # Use KernelExplainer (TreeExplainer is unstable for XGBoost in SHAP)
        explainer = shap.KernelExplainer(
            lambda x: ml_model.predict_proba(x)[:, 1],  # churn probability only
            bg_scaled_df
        )

        # Precompute SHAP values → improves UI speed
        shap_values_bg = explainer.shap_values(bg_scaled_df)
        logger.info("SHAP explainer loaded and SHAP values precomputed")

    except Exception as e:
        logger.warning("SHAP explainability disabled: %s", e)
        st.info("ℹ SHAP explainability is disabled due to limited resources or model issues.")
        explainer, bg_scaled_df, shap_values_bg = None, None, None

    # Final return
    return ml_model, dl_model, scaler, tfidf, df_featured, explainer, bg_scaled_df, shap_values_bg

# Load all assets
(ml_model, dl_model, scaler, tfidf, df_featured, explainer, bg_scaled_df, shap_values_bg) = load_assets()
logger.info("All assets loaded successfully — ready for prediction & dashboards")

    
# ROLE-BASED NAVIGATION (🔐 Access Control System)
# Sidebar Role Selector
st.sidebar.write("### 🔐 Select Your Role")
try:
    role = st.sidebar.selectbox(
        "Choose your access level:",
        [
            "👤 Customer / Basic User", "👨‍💼 Business / CRM Manager", "🧠 Data Scientist / Analyst",
        ],
    )
    logger.info(f"User selected role: {role}")
except Exception as e:
    st.error("⚠ Error loading roles — please refresh the page.")
    logger.error(f"Role selection error: {e}")
    st.stop()

# Define Role-based Pages
pages = {
    "👤 Customer / Basic User": [
        "🔍 Single Prediction",
        "📁 insights & history center",
        "ℹ Help & Info",
    ],
    "👨‍💼 Business / CRM Manager": [
        "📂 Batch Prediction",
        "📊 Business Dashboard",
        "📁 insights & history center",
        "ℹ Help & Info",
    ],
    "🧠 Data Scientist / Analyst": [
        "🔍 Single Prediction",
        "📂 Batch Prediction",
        "📊 Business Dashboard",
        "🧠 Explainability (SHAP)",
        "📁 insights & history center",
        "ℹ Help & Info",
    ],
}
# Sidebar Navigation – dynamic based on user role
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Prediction & Insights Hub")
st.sidebar.caption("📊 AI-powered decisions based on real data")

# Security: Fetch only allowed pages for selected role
try:
    page = st.sidebar.radio("🧭 Select a Page", pages[role])
    logger.info(f"Navigated to page: {page}")
except Exception as e:
    st.error("⚠ Page navigation failed — try refreshing.")
    logger.error(f"Page navigation error: {e}")
    st.stop()

# App Tittle
st.title("📡 Multimodal AI for Telecom Customer Churn & Sentiment Analysis")
st.write(
    "An end-to-end **Multimodal AI system** combining customer behavior & feedback "
    "to **predict churn & understand customer sentiment** for real-world CRM use cases."
)

# Footer Version Info
st.sidebar.markdown("---")
st.sidebar.caption("🧪 Model Versions: XGBoost v3 | DL v2 | Last Updated: Nov 2025")


# PAGE ROUTING– MAIN APP NAVIGATION  (ROLE BASED PREDICTION & DASHBOARDS)
# 1. SINGLE PREDICTION PAGE 
if page == "🔍 Single Prediction":

    #  ROLE VALIDATION – Only Customer & Data Scientist allowed
    if role not in ["👤 Customer / Basic User", "🧠 Data Scientist / Analyst"]:
        st.error("⛔ Access Denied! This page is only for *Customers & Data Scientists.*")
        logger.warning(f"Unauthorized access attempt to Single Prediction by {role}")
        st.stop()

    st.header("🔍 Single Customer Prediction – Churn & Feedback Sentiment Analysis")
    logger.info("Navigated to Single Prediction Page")

    # USER INPUT FORM (Structured + Unstructured)
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📥 Structured Customer Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_id = st.text_input("Customer ID")
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            SeniorCitizen = st.selectbox("Senior Citizen?", ["Yes", "No"])
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            monthly_charges = st.number_input("Monthly Charges", 0.0, 2000.0, 60.0)
            Partner = st.selectbox("Partner?", ["Yes", "No"])
        with c3:
            total_charges = st.number_input("Total Charges", 0.0, 100000.0, 600.0)
            is_high_value_customer = st.selectbox("High-Value Customer?", ["High Value", "Regular"])
            Dependents = st.selectbox("Dependents?", ["Yes", "No"])

        st.subheader("⚙️ Account & Billing")
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
        is_new_customer = st.selectbox("New Customer?", ["New", "Existing"])

        st.subheader("💬 Optional: Customer Feedback")
        feedback_text = st.text_area(
            "Customer Feedback (optional)",
            placeholder="Example: Service is okay, but billing is confusing and internet is slow...",
        )
   
    # Help & Usage Section (Right Side)
    with col_right:
        st.info(
            "💡 **What you can do here:**\n"
            "- Check churn for a **single customer**\n"
            "- Test **What-if Scenarios** by changing inputs\n"
            "- Add feedback 👉 sentiment analysis will run\n"
            "- Works like a real CRM retention tool 🚀"
        )
    
    # RUN PREDICTION BUTTON
    if st.button("🔮 Run Prediction"):
        try:
            logger.info("Running single prediction...")
            # Raw input (feature engineering happens in run_single_prediction)
            input_data = {
                "customerID": customer_id,
                "customer_id": customer_id,
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "Contract": Contract,
                "InternetService": InternetService,
                "PaymentMethod": PaymentMethod,
                "PaperlessBilling": PaperlessBilling,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "is_high_value_customer": is_high_value_customer,
                "is_new_customer": is_new_customer,
                "CustomerFeedback": feedback_text,
            }

            # Run predictions
            result, df_single, X_scaled = run_single_prediction(input_data, ml_model, dl_model, tfidf, scaler)
            if result is None:
                st.error("❌ Prediction could not be completed.")
                st.stop()

            add_single_history(input_data, result, conn, cursor)  # Save to DB + session history
            st.session_state["last_single_input"] = input_data  # Required for SHAP Explainability

            # PREDICTION SUMMARY
            st.markdown("### 🎯 Prediction Summary")
            colA, colB, colC = st.columns(3)
            colA.metric("🔎 Churn Prediction", result["churn_label"])
            colB.metric("📉 Churn Probability", f"{result['churn_probability']*100:.1f}%")
            colC.metric("💬 Sentiment", result["sentiment_label"])

            # Sentiment Display
            if feedback_text.strip():
                st.markdown("### 💬 Sentiment Details")
                st.write(f"**Predicted Sentiment:** {result['sentiment_label']}")
            else:
                st.info("⚠ No feedback provided — sentiment skipped.")
          
            logger.info("Single prediction completed successfully.")

        except Exception as e:
            st.error(f"❌ Unexpected error occurred: {e}")
            st.exception(e)  # detailed traceback for debugging


# 2. BATCH PREDICTION PAGE (CSV Upload)
elif page == "📂 Batch Prediction":
    # ROLE VALIDATION – Only Business + Data Scientist Allowed
    if role not in ["👨‍💼 Business / CRM Manager", "🧠 Data Scientist / Analyst"]:
        st.error("⛔ Access Denied! Only Business Managers & Data Scientists can run batch predictions.")
        logger.warning(f"Unauthorized access attempt to Batch Prediction by {role}")
        st.stop()

    st.header("📂 Batch Prediction – Upload CSV for Multiple Customers")
    logger.info("Navigated to Batch Prediction Page")

    st.write(
        "Upload a CSV with required feature columns. "
        "`CustomerFeedback` is optional for sentiment inference."
    )

    # SAMPLE CSV FORMAT DOWNLOAD
    st.markdown("📄 **Download Sample CSV Format Before Uploading:**")
    template_path = "data/templates/telco_sample_template_batch.csv"
    
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            st.download_button(
                label="📥 Download CSV Template",
                data=f,
                file_name="telco_sample_template.csv",
                mime="text/csv"
            )
    else:
        st.warning("CSV template not found. Please add it in  data/templates/")
    
    # FILE UPLOAD – Required for this page
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.markdown("#### 🔎 Preview of uploaded data")
        st.dataframe(df_input.head())

        # RUN BATCH PREDICTION
        if st.button("🚀 Run Batch Prediction"):
            try:
                logger.info(f"Running batch prediction on file: {uploaded_file.name}")
                df_scored = run_batch_predictions(df_input, scaler, ml_model, dl_model, tfidf) # ML + DL + Vader
                
                if df_scored.empty:
                    st.error("❌ Batch prediction returned no records.")
                    st.stop()

                add_batch_history(df_scored, cursor, conn, filename=uploaded_file.name)
                st.session_state["last_batch_df"] = df_scored.copy()

                st.success("✅ Batch prediction completedsuccessfully.")
                st.dataframe(df_scored.head())

                csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Prediction Results",
                    data=csv_bytes,
                    file_name="telco_churn_predictions.csv",
                    mime="text/csv",
                )
                logger.info("Batch prediction completed and file ready for download.")

            except Exception as e:
                st.error(f"❌ Error during batch prediction: {e}")
                st.exception(e)  # full traceback for debugging


# 3. BUSINESS DASHBOARD – Full Dataset + Batch Insights
elif page == "📊 Business Dashboard":
    # Dashboard is for Business Managers & Data Scientists
    if role not in ["👨‍💼 Business / CRM Manager", "🧠 Data Scientist / Analyst"]:
        st.error("⛔ Access Denied! This dashboard is only for CRM Managers & Data Scientists.")
        logger.warning(f"Unauthorized access attempt to Business Dashboard by {role}")
        st.stop()

    st.header("📊 Business Dashboard – Churn & Revenue Insights")
    logger.info("Navigated to Business Dashboard Page")

    # Dataset Upload (Mandatory)
    st.markdown("### 📂 Upload Custom Dataset to Generate dashboard Insights or use sample dataset.")

    # Upload Section (dataset must be provided by user)
    custom_dataset = st.file_uploader("📥 Upload CSV file for Business Dashboard", type=["csv"])

    if custom_dataset:
        try:
            df_featured = pd.read_csv(custom_dataset)
            st.success("✔ Dataset loaded successfully! Business Dashboard updated.")
            logger.info(f"Dataset uploaded: {custom_dataset.name}")

            # Try to enrich dataset with Churn Probability & Label
            df_temp = df_featured.copy()

            # Ensure numeric columns exist before processing
            for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce").fillna(0)

            # Generate churn predictions automatically if model is available
            df_temp[NUMERIC_COLS] = df_temp[NUMERIC_COLS].astype(float)
            X_scaled_temp = scaler.transform(df_temp[NUMERIC_COLS])
            churn_probs_temp = ml_model.predict_proba(X_scaled_temp)[:, 1]

            df_featured["Churn_Probability"] = np.round(churn_probs_temp, 3)
            df_featured["Churn_Label"] = np.where(churn_probs_temp >= 0.5, "Yes – Likely to Churn", "No – Safe Customer")
            st.dataframe(df_featured.head())
            logger.info("Churn probabilities added successfully.")

        except Exception as e:
            st.warning("⚠ Dataset loaded, but predictions could not be generated.")
            logger.error(f"Error generating churn prediction on dashboard dataset: {e}")
            pass      
        
    else:
        st.warning("⚠ No dataset uploaded — please upload your data to see business insights.")
        logger.info("Waiting for dataset upload for Business Dashboard.")

        # Download sample data (Business Dashboard Sample CSV)
        sample_csv = r"data\templates\telco_sample_template_business_dashboard.csv"
        if os.path.exists(sample_csv):
            with open(sample_csv, "rb") as f:
                st.download_button(
                    label="📥 Download Sample Dataset (ONLY for format reference)",
                    data=f.read(),
                    file_name="featured_telco_sample.csv",
                    mime="text/csv"
                )

        # Stop execution until user uploads data
        st.stop()

    # BUSINESS KPIs – Only if required columns are available
    st.markdown("---")
    st.subheader("📊 Key Business Metrics")

    if {"Churn", "MonthlyCharges", "tenure"}.issubset(df_featured.columns):
        try:
            churn_rate = df_featured["Churn"].mean() * 100
            avg_charges = df_featured["MonthlyCharges"].mean()
            avg_tenure = df_featured["tenure"].mean()
            revenue_at_risk = (df_featured["MonthlyCharges"] * df_featured["Churn"]).sum()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if churn_rate is not None:
                    st.metric("🔥 Churn Rate", f"{churn_rate:.1f}%")
            with col2:
                if avg_charges is not None:
                    st.metric("💵 Avg Monthly Charges", f"${avg_charges:.2f}")
            with col3:
                if avg_tenure is not None:
                    st.metric("📅 Avg Tenure (months)", f"{avg_tenure:.1f}")
            with col4:
                if revenue_at_risk is not None:
                    st.metric("💰 Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")
            logger.info("Business KPIs computed successfully.")
            
        except Exception as e:
            st.error("⚠ Error computing business KPIs.")
            logger.error(f"Error in KPI computation: {e}")
    else:
        st.warning("❗ Required columns not found: Churn, MonthlyCharges, tenure")
        logger.warning("Required KPI columns missing in dataset.")
    
    
    # KEY VISUALIZATIONS
    st.markdown("---")
    st.subheader("📈 Dataset Visualizations")
    colA, colB = st.columns(2)

    # Churn by Contract Type
    with colA:
        st.markdown("### 📌 Churn by Contract Type")
        if {"Contract", "Churn"}.issubset(df_featured.columns):
            try:
                # Map numeric → Text labels
                contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
                df_featured["Contract_Label"] = df_featured["Contract"].map(contract_map)

                fig, ax = plt.subplots(figsize=(6, 4))
                pd.crosstab(df_featured["Contract_Label"], df_featured["Churn"]).plot(kind="bar", ax=ax)
                ax.set_xlabel("Contract Type")
                ax.set_ylabel("Customer Count")
                ax.set_title("Churn Count by Contract Type")
                ax.tick_params(axis='x', rotation=0)
                st.pyplot(fig)
                plt.close(fig)
                logger.info("Contract-wise churn visualization completed.")
            except Exception as e:
                st.error("⚠ Error displaying Contract vs Churn.")
                logger.error(e)
        else:
            st.info("🛑 `Contract` or `Churn` column missing.")

    # Monthly Charges vs Churn        
    with colB:
        st.markdown("### 💸 Monthly Charges vs Churn")
        if {"MonthlyCharges", "Churn"}.issubset(df_featured.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            df_featured.boxplot(column="MonthlyCharges", by="Churn", ax=ax)
            ax.set_xlabel("Churn")
            ax.set_ylabel("Monthly Charges")
            ax.set_title("Monthly Charges by Churn Outcome")
            plt.suptitle("")
            st.pyplot(fig)
        else:
            st.info("MonthlyCharges or Churn column not found.")

    # Tenure bands & heatmap
    colC, colD = st.columns(2)
    with colC:
        st.markdown("### ⏳ Churn by Tenure Band")
        if {"tenure", "Churn"}.issubset(df_featured.columns):
            df_temp = df_featured.copy()
            df_temp["TenureBand"] = pd.cut(
                df_temp["tenure"],
                bins=[0, 12, 24, 36, 48, 60, 100],
                labels=["0–12", "12–24", "24–36", "36–48", "48–60", "60+"],
                right=False,
            )
            churn_by_tenure = df_temp.groupby("TenureBand", observed=False)["Churn"].mean() * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            churn_by_tenure.plot(kind="bar", ax=ax)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xlabel("Tenure Band (months)")
            ax.set_title("Churn Rate by Tenure Band (%)")
            st.pyplot(fig)
        else:
            st.info("tenure or Churn column not found in dataset.") 
   
    # Heatmap – Tenure vs Monthly Charges 
    with colD:      
        st.markdown("### 🔥 Churn Risk Heatmap – Tenure vs Monthly Charges")

        if {"tenure", "MonthlyCharges", "Churn"}.issubset(df_featured.columns):
            try:
                df_temp = df_featured.copy()

                # Create Tenure Bands
                df_temp["TenureBand"] = pd.cut(
                    df_temp["tenure"], 
                    bins=[0,12,24,36,48,60,100], 
                    labels=["0–12","12–24","24–36","36–48","48–60","60+"]
                )

                # Create Monthly Charges Groups 
                df_temp["ChargeBand"] = pd.cut(
                    df_temp["MonthlyCharges"],
                    bins=[0,40,80,200],   # You can modify ranges
                    labels=["Low", "Medium", "High"]
                )

                # Pivot Table
                pivot = df_temp.pivot_table(
                    values="Churn",
                    index="TenureBand",
                    columns="ChargeBand",
                    aggfunc="mean",
                    observed=False 
                )

                # Plot Heatmap
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(pivot, aspect="auto")
                ax.set_title("Heatmap – Churn Risk (Tenure vs Charges)")
                ax.set_xlabel("Customer Spending Level")
                ax.set_ylabel("Tenure Band (months)")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_yticks(range(len(pivot.index)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticklabels(pivot.index)
                st.pyplot(fig)

                plt.close(fig)
                plt.close("all")
                logger.info("Heatmap visualization generated.")
            
            except Exception as e:
                st.error("⚠ Error generating heatmap.")
                st.exception(e)

        else:
            st.info("Required columns not found in dataset.")

    
    #  ADVANCED BUSINESS INSIGHTS
    st.markdown("---")
    st.markdown("## 🧠 Advanced Business Insights")

    # CLV – Customer Lifetime Value 
    st.markdown("### 💰 Customer Lifetime Value (CLV)")
    if {"tenure", "MonthlyCharges"}.issubset(df_featured.columns):
        df_featured["CLV"] = df_featured["tenure"] * df_featured["MonthlyCharges"]
        
        st.write("**Avg CLV:**", round(df_featured["CLV"].mean(), 2))
        fig, ax = plt.subplots(figsize=(6, 4))
        df_featured["CLV"].hist(ax=ax, bins=30)
        ax.set_title("CLV Distribution", fontsize=12)
        ax.set_xlabel("Customer Lifetime Value", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        st.pyplot(fig)
        plt.close(fig)

    colE, colF = st.columns(2)
    with colE:
        # Revenue Segmentation 
        st.markdown("### 🧾 Revenue Segmentation")
        if "MonthlyCharges" in df_featured.columns:
            df_featured["RevenueSegment"] = pd.cut(df_featured["MonthlyCharges"], bins=[0, 40, 80, 200], labels=["Low Value", "Regular", "VIP"])
            seg_counts = df_featured["RevenueSegment"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            seg_counts.plot(kind="bar", ax=ax)
            ax.set_title("Revenue Segment Distribution", fontsize=12)
            ax.set_xlabel("Customer Revenue Category", fontsize=11)
            ax.set_ylabel("Number of Customers", fontsize=11)
            st.pyplot(fig)
            plt.close(fig)

    with colF:
        # Churn Risk Segmentation 
        st.markdown("### ⚠ Churn Risk Segmentation")
        if "Churn" in df_featured.columns:
            df_featured["RiskZone"] = pd.cut(
                df_featured["Churn_Probability"] if "Churn_Probability" in df_featured.columns else df_featured["Churn"],
                bins=[0, 0.4, 0.7, 1],
                labels=["🟢 Low Risk", "⚠ Medium Risk", "🔥 High Risk"]
            )
            risk_counts = df_featured["RiskZone"].value_counts()

            fig, ax = plt.subplots(figsize=(5, 4))
            risk_counts.plot(kind="bar", ax=ax)
            ax.set_title("Churn Risk Distribution", fontsize=12)
            ax.set_xlabel("Risk Category", fontsize=11)
            ax.set_ylabel("Number of Customers", fontsize=11)
            st.pyplot(fig)
            plt.close(fig)

    # Retention Strategy Recommendation 
    st.markdown("### 🛠 Retention Strategy Recommendation")

    def get_strategy(churn_prob, sentiment):
        if churn_prob > 0.7 and sentiment == "Negative":
            return "🛑 Immediate Customer Support & Discount Offer"
        elif churn_prob > 0.5 and sentiment in ["Neutral", "Negative"]:
            return "📞 Call & Explain Billing / Offers"
        elif churn_prob < 0.4:
            return "🟢 Send Engagement Email"
        else:
            return "⚠ Monitor Behavior"

    # Check if churn probability is already available or not
    if "Churn_Probability" not in df_featured.columns:
        st.warning("Churn_Probability column not found. Run batch prediction first.")
    else:
        if "Sentiment_Label" not in df_featured.columns:
            df_featured["Sentiment_Label"] = "Neutral"   # fallback

        # Apply strategy to each row
        df_featured["Retention_Strategy"] = df_featured.apply(
            lambda x: get_strategy(x["Churn_Probability"], x["Sentiment_Label"]), axis=1
        )

        st.dataframe(
            df_featured[
                ["tenure", "MonthlyCharges", "Churn_Probability", "Sentiment_Label", "Retention_Strategy"]
            ].head(20)  # show first 20
        )

        # Strategy count plot
        strategy_counts = df_featured["Retention_Strategy"].value_counts()
        st.markdown("### 📊 Strategy Distribution")
        st.bar_chart(strategy_counts)

        # Suggested action
        st.markdown("#### 📍 Recommendation:")
        st.write(f"➡ Most frequent strategy: **{strategy_counts.idxmax()}**")
    
    # Summary & PDF (Business Report)
    st.markdown("---")
    st.subheader("📌 Business Summary & Report Generator")

    if {"Churn", "MonthlyCharges"}.issubset(df_featured.columns):
        total_customers = len(df_featured)
        high_risk = (df_featured["Churn"] == 1).sum()
        avg_revenue_risk = (df_featured["MonthlyCharges"] * df_featured["Churn"]).mean()

        neg_feedback_pct = 0
        if "Sentiment_Label" in df_featured.columns:
            neg_feedback_pct = (df_featured["Sentiment_Label"] == "Negative").mean() * 100

        summary_df = pd.DataFrame({
            "Metric": [
                "Total Customers",
                "High Churn Risk Customers",
                "Avg Revenue at Risk ($)",
                "Negative Feedback %"
            ],
            "Value": [
                str(total_customers),
                str(high_risk),
                f"${round(avg_revenue_risk, 2)}",
                f"{neg_feedback_pct:.2f}%"    
            ]
        })

        summary_df = summary_df.astype(str)
        st.table(summary_df)
    else:
        st.warning("Data is missing some required columns for summary insights.")

    # PDF Download Button 
    if st.button("📥 Download Business Insights Report"):
            try:
                file_path = generate_business_report(summary_df, company_name="TelcoWave Solutions Pvt Ltd")
                st.success(f"Report saved! Click below to download.")
                with open(file_path, "rb") as f:
                    st.download_button("📄 Download PDF Report", f, file_name="business_insights_report.pdf")
                logger.info("PDF Report generated successfully.")
            except Exception as e:
                st.error("⚠ PDF generation failed.")
                logger.error(e)


# 4. SHAP MODEL EXPLAINABILITY
elif page == "🧠 Explainability (SHAP)":
    # ROLE VALIDATION – STRICT ACCESS CONTROL
    if role != "🧠 Data Scientist / Analyst":
        st.error("⛔ Access Denied! SHAP Explainability is for Data Scientists only.")
        logger.warning(f"Unauthorized access attempt to SHAP Page by role: {role}")
        st.stop()

    st.header("🧠 Model Explainability – SHAP Insights")
    logger.info("Navigated to SHAP Explainability Page")

    # TABS – LOCAL / BATCH / GLOBAL LEVEL EXPLAINABILITY
    exp_tabs = st.tabs(
        [
            "🔍 Single Prediction SHAP",
            "📂 Batch Prediction SHAP",
            "🌍 Global Dataset SHAP",
        ])

    # TAB 1: LOCAL EXPLAINABILITY → Single Prediction
    with exp_tabs[0]:
        st.subheader("🔍 Single Prediction Explainability")
        if st.session_state["last_single_input"] is None:
            st.info("⚠ No single prediction available. Run a prediction first.")
            logger.info("No last_single_input found for SHAP Single Explainability")
        else:
            try:
                plot_single_shap_from_input(st.session_state["last_single_input"], explainer, scaler)
                st.success("✔ Local explainability insights generated successfully.")
                logger.info("Single SHAP explainability plotted successfully.")
            except Exception as e:
                st.error("❌ Error generating single prediction SHAP.")
                st.exception(e)
                logger.error(f"Error in single SHAP: {e}")

    # TAB 2: BATCH EXPLAINABILITY → Customer Group Analysis
    with exp_tabs[1]:
        st.subheader("📂 Batch Explainability & Comparison")
        df_batch = st.session_state.get("last_batch_df", None)
        
        if df_batch is None:
            st.info("⚠ No batch predictions available. Upload a CSV and run batch prediction.")
            logger.info("No last_batch_df found for batch SHAP explainability")
        else:
            try:
                # Primary batch-level SHAP
                plot_batch_shap_from_df(df_batch,  explainer, scaler)
                logger.info("Batch SHAP explainability plotted successfully.")

                # Compare churn vs non-churn only if label exists
                if "Churn_Label" in df_batch.columns:
                    st.markdown("---")
                    plot_churn_vs_nonchurn_shap(df_batch,  explainer, scaler)
                    logger.info("Churn vs non-churn SHAP comparison plotted.")
                else:
                    st.warning("⚠ Column `Churn_Label` not found — cannot compare churn groups.")
                    logger.warning("Churn_Label missing in batch dataframe.")
            except Exception as e:
                st.error("❌ Error generating batch SHAP explainability.")
                st.exception(e)
                logger.error(f"Error in batch SHAP: {e}")

    # TAB 3: GLOBAL EXPLAINABILITY → Dataset-Level Insights
    with exp_tabs[2]:
        st.subheader("🌍 Global Explainability – Across Dataset")
        try:
            plot_global_shap(explainer, shap_values_bg, bg_scaled_df)  # main global explainability
            logger.info("Global SHAP summary plotted successfully.")
        except Exception as e:
            st.error("❌ Error computing global SHAP summary.")
            st.exception(e)
            logger.error(f"Error in global SHAP: {e}")

        # Auto-generated feature insights
        shap_vals = _select_positive_class_shap(shap_values_bg)
        if shap_vals is not None:
            try:
                generate_feature_insights(shap_vals, NUMERIC_COLS)
                logger.info("Auto Feature Insights generated successfully")
            except Exception as e:
                st.warning("⚠ Unable to generate feature insights.")
                logger.error(f"Feature insight generation failure: {e}")

# 5. UNIFIED INSIGHTS & HISTORY CENTER
elif page == "📁 insights & history center":
    # ROLE CHECK
    if role not in ["👤 Customer / Basic User", "👨‍💼 Business / CRM Manager", "🧠 Data Scientist / Analyst"]:
        st.error("⛔ Access Denied! You do not have permission to view this page.")
        logger.warning(f"Unauthorized access detected: {role}")
        st.stop()

    st.header("📁 Unified Insights & History Center")
    st.caption("📌 View, Analyze & Download Past Predictions — Session + Database")
    logger.info("Navigated to Insights & History Center Page")

    # TABS FOR DIFFERENT HISTORY TYPES
    if role == "👤 Customer / Basic User":
        tabs = st.tabs(["🔍 Single History"])
    elif role == "👨‍💼 Business / CRM Manager":
        tabs = st.tabs(["🔍 Single History", "📂 Batch History", "📊 Summary Insights"])
    else:  # Data Scientist
        tabs = st.tabs(["🔍 Single History", "📂 Batch History", "📊 Summary Insights", "💾 Backup / Restore"])

    # TAB 1 – SINGLE PREDICTION HISTORY (SESSION + DB)
    with tabs[0]:
        st.subheader("🔍 Single Prediction History (Session-Level)")
        try:
            single_hist = st.session_state["history"]["single"]
            if single_hist:
                df_hist_single = pd.DataFrame(single_hist)
                st.dataframe(df_hist_single)

                # Download History 
                csv_single = df_hist_single.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Single History ",
                    data=csv_single,
                    file_name="single_history.csv",
                    mime="text/csv",
                )

                # KPI Metrics 
                try:
                    avg_prob = df_hist_single["churn_probability"].mean() * 100
                    total_predictions = len(df_hist_single)
                    churn_yes = (df_hist_single["churn_label"] == "Yes – Likely to Churn").sum()
                    churn_no = (df_hist_single["churn_label"] == "No – Safe Customer").sum()

                    st.markdown("---")
                    st.subheader("📊 📌 Key Performance Indicators")
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Predictions", total_predictions)
                    col2.metric("Churn – YES", churn_yes)
                    col3.metric("Churn – NO", churn_no)
                    col4.metric("Avg Churn Probability", f"{avg_prob:.2f}%")
                except Exception as e:
                    st.warning("⚠ Unable to compute session KPIs")
                    logger.error(f"Error computing single KPIs: {e}")
                
                # Visualizations 
                st.markdown("---")
                st.markdown("### 🔢 Churn Decision Breakdown")
                st.bar_chart(df_hist_single["churn_label"].value_counts())

                st.markdown("### 📌 Probability Spread (Distribution)")
                st.bar_chart(df_hist_single["churn_probability"])

            else:
                st.info("No single predictions made in this session yet.")

            # Show DB history (persistent). CUSTOMER SHOULD NOT SEE DB HISTORY
            if role == "👤 Customer / Basic User":
                st.warning("🔒 Database history is restricted for basic users.")
            else:
                st.markdown("---")
                st.subheader("📦 Database: Saved Single Predictions")
                df_db_single = pd.read_sql("SELECT * FROM predictions WHERE mode='single'", conn)
                
                if df_db_single.empty:
                    st.info("No single prediction records found in database.")
                else:
                    st.dataframe(df_db_single, width="stretch") 

        except Exception as e:
            st.error("❌ Something went wrong loading single history.")
            logger.error(f"Error in Single History Tab: {e}")


    # TAB 2 – BATCH HISTORY (SESSION + DB)
    if role != "👤 Customer / Basic User":
        with tabs[1]:
            st.subheader("📂 Batch Prediction History (Session-Level)")
            try:
                batch_hist = st.session_state["history"]["batch"]
                df_batch = st.session_state.get("last_batch_df", None)

                if batch_hist:
                    df_hist_batch = pd.DataFrame(batch_hist)
                    st.dataframe(df_hist_batch)

                    # Download session history
                    csv_batch = df_hist_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Session Batch History",
                        data=csv_batch,
                        file_name="batch_history.csv",
                        mime="text/csv",
                    )

                else:
                    st.info("ℹ No batch predictions made in this session yet.")

                # KPI SECTION 
                st.markdown("---")
                st.subheader("📊 📌 Key Performance Indicators")
                col1, col2 = st.columns(2)

                # Total batch runs (session)
                total_batches = len(batch_hist) if batch_hist else 0
                col1.metric("Total Batch Runs", total_batches)

                # Avg churn rate across batches (session-level df_batch)
                col2.metric("Avg Churn Rate (Last Batch)", 
                        f"{df_batch['Churn_Probability'].mean()*100:.2f}%" if df_batch is not None else "⚠ No Data")

                # VISUALS 
                if df_batch is None:
                    st.info("⚠ No batch predictions in this session to display KPIs.")
                else:
                    st.markdown("---")          
                    # Churn label distribution
                    st.markdown("### 🔢 Churn Decision Breakdown")
                    st.bar_chart(df_batch["Churn_Label"].value_counts())

                    # Sentiment distribution (if exists)
                    if "Sentiment_Label" in df_batch.columns:
                        st.markdown("### 💬 Sentiment Distribution")
                        st.bar_chart(df_batch["Sentiment_Label"].value_counts())

                # DATABASE HISTORY SECTION 
                st.markdown("---")
                st.subheader("📦 Database: Saved Batch Predictions")
                try:
                    df_db_batch = pd.read_sql("SELECT * FROM predictions WHERE mode='batch'", conn)
        
                    if df_db_batch.empty:
                        st.info("ℹ No batch data stored in DB yet.")
                    else:
                        st.dataframe(df_db_batch, width="stretch")    
            
                except Exception as e:
                    st.error("❌ Error loading batch records from database.")
                    logger.error(f"[Batch History] DB load error: {e}")

            except Exception as e:
                st.error("❌ Error loading batch history.")
                logger.error(f"Batch History Tab Error: {e}")

        
    # TAB 3 – SUMMARY DASHBOARD (DB-Level KPIs & Charts)
    if role != "👤 Customer / Basic User":
        with tabs[2]:
            st.subheader("📊 Summary Insights — Full DB Analysis")
            logger.info("Accessed Summary Insights Tab")

            try:
                df_db_all = pd.read_sql("SELECT * FROM predictions", conn)
            except Exception as e:
                st.error("❌ Could not load prediction history from database.")
                logger.error(f"[Summary Tab] DB load failure: {e}")
                df_db_all = pd.DataFrame()

            if df_db_all.empty:
                st.info("⚠ No data in database yet!")
            else:
                # HIGH LEVEL KPI CARDS
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📌 Total Predictions", len(df_db_all))

                avg_churn = df_db_all["churn_probability"].mean() * 100
                col2.metric("📉 Avg Churn Probability", f"{avg_churn:.2f}%")

                neg_feedback_pct = (df_db_all["sentiment_label"] == "Negative").mean() * 100
                col3.metric("🚩 Negative Feedback %", f"{neg_feedback_pct:.2f}%")

                churn_count = (df_db_all["churn_label"] == "Yes – Likely to Churn").sum()
                col4.metric("🔥 Total High Risk Customers", churn_count)

                # CHURN & SENTIMENT DISTRIBUTION
                st.markdown("---")
                st.markdown("### 📌 Churn Decision Breakdown (DB Data)")
                st.bar_chart(df_db_all["churn_label"].value_counts())

                if "sentiment_label" in df_db_all.columns:
                    st.markdown("### 💬 Sentiment Distribution")
                    st.bar_chart(df_db_all["sentiment_label"].value_counts())

                # MONTHLY CHURN TREND
                st.markdown("---")
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("### 📈 Monthly Churn Trend")

                    try:
                        df_db_all['timestamp'] = pd.to_datetime(df_db_all['timestamp'], errors='coerce')
                        df_db_all['Month'] = df_db_all['timestamp'].dt.to_period('M')

                        trend = df_db_all.groupby("Month")["churn_probability"].mean()
                        trend_df = trend.to_frame().reset_index()
                        trend_df["Month"] = trend_df["Month"].astype(str)

                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(trend_df["Month"], trend_df["churn_probability"], marker="o")
                        ax.set_title("Monthly Churn Trend (DB History)")
                        ax.set_xlabel("Month")
                        ax.set_ylabel("Avg Churn Probability")
                        ax.tick_params(axis='x', rotation=45)

                        st.pyplot(fig)
                        plt.close(fig)

                    except Exception as e:
                        st.warning("⚠ Could not generate time-series trend.")
                        logger.error(f"Time-series error: {e}")

                #  BASIC SEGMENT ANALYSIS (Value × New Customer)
                with col2:
                    st.subheader("🎯 Customer Segment-wise Analysis")

                    if {"is_high_value_customer", "is_new_customer"}.issubset(df_db_all.columns):
                        try:
                            df_seg_basic = df_db_all.copy()

                            df_seg_basic["value_segment"] = df_seg_basic["is_high_value_customer"].astype(str).str.title()
                            df_seg_basic["customer_type"] = df_seg_basic["is_new_customer"].astype(str).str.title()

                            seg_counts = df_seg_basic.groupby(["value_segment", "customer_type"]).size()
                            seg_counts_df = seg_counts.unstack(fill_value=0)

                            fig, ax = plt.subplots(figsize=(7, 5))
                            seg_counts_df.plot(kind="bar", ax=ax)

                            ax.set_title("Customer Segment Breakdown (Value Segment vs.Type)", fontsize=12)
                            ax.set_xlabel("Customer Value Segment")
                            ax.set_ylabel("Number of Customers")
                            ax.legend(title="Customer Type")

                            st.pyplot(fig)
                            plt.close(fig)

                        except Exception as e:
                            st.warning("⚠ Could not compute segment breakdown.")
                            logger.error(f"[Summary Tab] Basic Segmentation error: {e}")
                    else:
                        st.warning("Segmentation columns missing. Run batch prediction first.")

                # SEGMENT ANALYSIS (ADVANCED ANALYTICS)
                st.markdown("---")
                st.subheader("🎯 Customer Segment Analysis (Value × New Customer × Churn × Sentiment)")

                required_cols = {"is_high_value_customer", "is_new_customer", "churn_probability", "churn_label"}
                if required_cols.issubset(df_db_all.columns):

                    try:
                        df_seg = df_db_all.copy()

                        # Label Cleaning
                        df_seg["value_segment"] = df_seg["is_high_value_customer"].astype(str).str.title()
                        df_seg["customer_type"] = df_seg["is_new_customer"].astype(str).str.title()
                   
                        col1, col2 = st.columns(2)
                        with col1:
                            #  POPULATION COUNT
                            st.markdown("### 1️⃣ Segment Population Count")
                            seg_counts = df_seg.groupby(["value_segment", "customer_type"]).size()
                            seg_counts_df = seg_counts.unstack(fill_value=0)

                            fig, ax = plt.subplots(figsize=(7, 5))
                            seg_counts_df.plot(kind="bar", ax=ax)
                            ax.set_title("Customer Population by Segment")
                            ax.set_xlabel("Customer Value Segment")
                            ax.set_ylabel("Number of Customers")
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                            plt.close(fig)

                        # CHURN RATE PER SEGMENT
                        with col2:
                            st.markdown("### 2️⃣ Churn Rate per Segment (%)")
                            churn_rates = (
                                df_seg.groupby(["value_segment", "customer_type"])["churn_label"]
                                .apply(lambda x: (x == "Yes – Likely to Churn").mean() * 100)
                                .unstack(fill_value=0)
                            )

                            fig, ax = plt.subplots(figsize=(7, 5))
                            churn_rates.plot(kind="bar", ax=ax)
                            ax.set_title("Churn Rate (%)")
                            ax.set_xlabel("Customer Value Segment")
                            ax.set_ylabel("Churn Percentage (%)")
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                            plt.close(fig)

                        # Avg Churn Probability
                        col3, col4 = st.columns(2)
                        with col3:
                            st.markdown("### 3️⃣ Avg Churn Probability per Segment")
                            avg_probs = (
                                df_seg.groupby(["value_segment", "customer_type"])["churn_probability"]
                                .mean()
                                .unstack(fill_value=0)
                            )

                            fig, ax = plt.subplots(figsize=(7, 5))
                            avg_probs.plot(kind="bar", ax=ax)
                            ax.set_title("Avg Churn Probability")
                            ax.set_xlabel("Customer Value Segment")
                            ax.set_ylabel("Probability")
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                            plt.close(fig)

                        # Revenue at Risk
                        with col4:
                            st.markdown("### 4️⃣ Revenue at Risk per Segment")
                            if "MonthlyCharges" in df_seg.columns:
                                df_seg["revenue_risk"] = df_seg["MonthlyCharges"] * df_seg["churn_probability"]

                                rev_risk = (
                                    df_seg.groupby(["value_segment", "customer_type"])["revenue_risk"]
                                    .sum()
                                    .unstack(fill_value=0)
                                )

                                fig, ax = plt.subplots(figsize=(7, 5))
                                rev_risk.plot(kind="bar", ax=ax)
                                ax.set_title("Revenue at Risk")
                                ax.set_xlabel("Customer Value Segment")
                                ax.set_ylabel("Revenue at Risk ($)")
                                plt.xticks(rotation=0)
                                st.pyplot(fig)
                                plt.close(fig)

                        # Sentiment Distribution
                        if "sentiment_label" in df_seg.columns:
                            st.markdown("### 5️⃣ Sentiment Distribution per Segment")
                            sent_counts = (
                                df_seg.groupby(["value_segment", "customer_type", "sentiment_label"])
                                .size()
                                .unstack(fill_value=0)
                            )
                            st.dataframe(sent_counts)

                            st.success("✨ PRO Segment Analysis Completed!")

                    except Exception as e:
                        st.warning("⚠ PRO Segment Analysis Failed.")
                        logger.error(f"[PRO Segment Error] {e}")

                # AI INSIGHT GENERATOR
                st.markdown("---")
                st.subheader("🤖 AI Insights Generator")
                churn_rate = (df_db_all["churn_label"]=="Yes – Likely to Churn").mean()*100
                neg_sent = (df_db_all["sentiment_label"]=="Negative").mean()*100

                st.write(f"📌 **High churn rate → {churn_rate:.1f}%**")
                st.write(f"📌 **Negative sentiment → {neg_sent:.1f}%**")

                if churn_rate > 50:
                    st.warning("⚠ Critical churn risk! Immediate action needed.")
                elif neg_sent > churn_rate:
                    st.info("💬 Sentiment is more negative than churn — focus on customer support!")
                else:
                    st.success("✔ Customer base appears stable.")

                # DOWNLOAD FULL DB HISTORY
                st.markdown("---")
                st.markdown("### 📥 Download Full Prediction History (DB Data)")
                csv_all = df_db_all.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Full History as CSV",
                    csv_all,
                    "full_prediction_history.csv",
                    mime="text/csv",
                )

    # TAB 4 – BACKUP / RESTORE / FILTER
    if role == "🧠 Data Scientist / Analyst":
        with tabs[3]:
            st.subheader("💾 Backup & Restore Prediction History")
            logger.info("Accessed Backup & Restore Tab")

            # Manual Backup
            if st.button("💾 Backup Current DB to CSV"):
                try:
                    backup_history_csv(conn)
                    st.success("🟢 Backup saved to outputs directory.")
                    logger.info("Manual DB backup triggered from Insights & History Center.")
                except Exception as e:
                    st.error("❌ Failed to backup database to CSV.")
                    logger.error(f"[Backup Tab] Backup error: {e}")

            st.markdown("---")
            # Restore
            st.markdown("### 🔄 Upload Old History to Restore DB")
            uploaded_history = st.file_uploader("Upload CSV History", type=["csv"])
            
            if uploaded_history:
                try:
                    df_restore = pd.read_csv(uploaded_history)
                    expected_cols =["timestamp","mode","customer_id","gender","SeniorCitizen","Partner","Dependents","Contract","InternetService","PaymentMethod","PaperlessBilling",
                                    "tenure","MonthlyCharges","TotalCharges","is_high_value_customer","is_new_customer","churn_label","churn_probability","sentiment_label"]
                    for col in expected_cols:
                        if col not in df_restore.columns:
                            df_restore[col] = None

                    df_restore[expected_cols].to_sql("predictions", conn, if_exists="append", index=False)
                    conn.commit()
                    st.success("📦 History restored into DB!")
                    logger.info("History CSV restored to DB")
                except Exception as e:
                    st.error("❌ Error restoring database.")
                    logger.error(f"Restore operation failed: {e}")

            # Filter DB History
            st.markdown("---")
            st.subheader("🔎 Filter DB Records")

            filter_sent = st.selectbox("Filter by Sentiment", ["All","Positive","Neutral","Negative"])
            filter_churn = st.selectbox("Filter by Churn", ["All","Yes – Likely to Churn","No – Safe Customer"])

            query = "SELECT * FROM predictions"
            conditions = []

            if filter_sent != "All":
                conditions.append(f"sentiment_label = '{filter_sent}'")
            if filter_churn != "All":
                conditions.append(f"churn_label = '{filter_churn}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            try:
                df_filtered = pd.read_sql(query, conn)
                st.dataframe(df_filtered, width="stretch")

                # Download filtered
                if not df_filtered.empty:
                    csv_filtered = df_filtered.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Filtered DB History", csv_filtered, "filtered_history.csv", mime="text/csv")   
            except Exception as e:
                st.error("❌ Failed to apply filters on DB history.")
                logger.error(f"[Backup Tab] Filter query error: {e}")
        
# 6. ℹ HELP & INFO – USER MANUAL
elif page == "ℹ Help & Info":
    st.header("ℹ Help & Info – Application Guide")
    logger.info("Navigated to Help & Info Page")

    # BUSINESS PROBLEM EXPLAIN
    st.markdown("""
    ### 📌 Real Business Problem – Why This App?
    Telecom companies lose **millions every year due to customer churn**.  
    This app helps by:
    ✔ Predicting **who is likely to churn**  
    ✔ Identifying **why they may leave**  
    ✔ Suggesting **business retention strategies**

    👉 This turns **data into action** → used in **CRM & customer retention teams**.
    """)

    st.info("💼 **Business Value = Reduce churn + improve satisfaction + retain revenue**")
    logger.info("Displayed business problem section")

    # PAGE NAVIGATION TABLE – PURPOSE OF EACH PAGE
    st.markdown("""
    ---  
    ### 🧭 Pages & Purpose (Quick Navigation)

    | Page | Purpose |
    |------|---------|
    | 🔍 Single Prediction | Predict churn + sentiment for one customer |
    | 📂 Batch Prediction | Upload CSV → Predict multiple customers |
    | 📊 Business Dashboard | CLV, segmentation, trends, risk zones |
    | 🧠 Explainability (SHAP) | Why did the model predict churn? |
    | 📁 Insights & History Center | View / filter / download history |
    | ℹ Help & Info | Project info (Interview + Resume Purpose) |
    """)
    logger.info("Displayed pages & purpose table")

    # TECH STACK – INDUSTRY STANDARD USED HERE
    st.markdown("""
    ---  
    ### 🧠 Technology Stack (Industry-Level Use Case)

    - **Machine Learning (XGBoost)** → Churn Prediction  
    - **Deep Learning (Neural Network)** → Sentiment Analysis  
    - **TF-IDF + NLP** → Text Feature Engineering  
    - **SHAP Explainability** → Feature Influence (Data Science Standard)  
    - **SQLite Database** → Permanent History Storage  
    - **Streamlit** → Frontend UI with Role-Based Access   
    - **Domain Use-Case** → Telecom | CRM | Customer Retention  
    """)
    logger.info("Displayed Tech Stack section")

    # MODEL PERFORMANCE – KPIs & BENCHMARKS
    st.markdown("### 🔬 Model Performance (Evaluation Metrics)")

    try:
        # ML MODEL METRICS TABLE
        ml_eval = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Test Accuracy": [0.8573, 0.9092, 0.9177],
            "ROC-AUC":       [0.9422, 0.9522, 0.9677],
            "F1 Score":      [0.8618, 0.9067, 0.9177],
            "Remark": ["✔ Good Fit", "✔ Good Fit", "🏆 Best Model"]
        })
        ml_eval["Final Score"] = ml_eval[['Test Accuracy', 'ROC-AUC', 'F1 Score']].mean(axis=1)

        st.write("#### 🧪 ML Models – Comparison")
        st.dataframe(ml_eval)
        st.success(f"🏆 Selected Final ML Model: **{ml_eval.loc[ml_eval['Final Score'].idxmax(), 'Model']}**")
    except Exception as e:
        logger.error(f"Error while displaying ML model performance: {e}")
        st.error("⚠ Error loading ML model metrics.")
    try:
        # DL MODEL METRICS    
        dl_eval = pd.DataFrame({
            "Model": ["Neural Network (DL)"],
            "Test Accuracy": [0.8815],
            "ROC-AUC":       [0.0],  # Not applicable
            "F1 Score":      [0.8777],
            "Remark": ["✔ Good Fit"]
        })
        dl_eval["Final Score"] = dl_eval[['Test Accuracy', 'F1 Score']].mean(axis=1)

        st.write("#### 🔶 DL Model – Sentiment Analysis")
        st.dataframe(dl_eval)
    except Exception as e:
        logger.error(f"Error while displaying DL model performance: {e}")
        st.error("⚠ Error loading DL model metrics.")

    st.info("""
    ✔ **XGBoost → Final Churn Model**  
    ✔ **Neural Network → Final Sentiment Model**  
    ✔ Both models can be deployed in real telecom CRM systems  
    """)
    logger.info("Displayed Model Performance section")

    # FUTURE ROADMAP
    st.markdown("""
    ---  
    ### 🚀 Future Enhancements (Industry Deployment)

    - API Deployment using **FastAPI / Flask**
    - Login System with **Firebase / OAuth**
    - Cloud Hosting (AWS / Render / Railway)
    - Customer Support **Chatbot using GenAI**
    - CRM Integration with **Salesforce / Zendesk/ Zoho**
    """)
    logger.info("Displayed project roadmap")

    # INTERNSHIP DETAILS
    st.markdown("""
    ---  
    ### 🏢 Internship Project Details (Resume / LinkedIn)

    | Field | Details |
    |------|---------|
    | Project Type | Industrial Internship Project |
    | Company | **InfozIT Solutions Pvt. Ltd.** |
    | Role | Data Science Intern |
    | Duration | 2025 |
    | Domain | Telecom – Churn & Customer Retention |
    | **Skills Used** | ML, DL, NLP, SHAP, SQL, Streamlit |
    """)
    logger.info("Displayed internship/resume-related content")

    st.markdown("""
    **Main Objectives:**
    ✔ Predict customer churn  
    ✔ Understand customer sentiment  
    ✔ Provide business-level insights & retention strategy  
    """)

    # ABOUT AUTHOR
    st.markdown("---")
    st.subheader("👩‍💻 About the Developer")

    st.write("""
    **Name:** *Ayesha Banu*  
    🏅 **Gold Medalist & 1st Rank in M.Sc. Computer Science**   
    🔍 Aspiring **Data Analyst / ML Engineer / Data Scientist Developer**  
    """)

    st.write("📧 **Email:** ayesha24banu@gmail.com")
    st.write("🔗 **LinkedIn:**  https://www.linkedin.com/in/ayesha_banu_cs")
    st.write("⭐ **GitHub:** *(Add your project repository here)*")

    st.sidebar.info("📌 Internship Project – InfozIT Solutions Pvt. Ltd. (2025)")
    logger.info("Displayed developer profile section")

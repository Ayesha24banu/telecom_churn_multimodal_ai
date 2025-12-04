# PATH CONFIG
import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_RAW = os.path.join(PROJECT_ROOT, "data/raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data/processed")
DATA_FEATURED = os.path.join(PROJECT_ROOT, "data/featured")  

# Models / Encoders / Scalers
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Reports / Dashboards
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
EDA_DIR = os.path.join(PROJECT_ROOT, "reports/eda_figures")

# Outputs / ML and DL model evaluation
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_VISUALS_DIR= os.path.join(OUTPUTS_DIR, "models_visuals")


# Logs
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_PATH = os.path.join(LOGS_DIR, "customer_churn_feedback.log")
APP_LOG_PATH = os.path.join(LOGS_DIR, "app_run.log")
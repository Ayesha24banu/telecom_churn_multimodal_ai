# 📡 Multimodal AI for Telecom Customer Churn & Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![TensorFlow](https://img.shields.io/badge/DL-TensorFlow-red)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)
![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![License](https://img.shields.io/badge/license-RA--License-blue)

### 🔍 An End-to-End Multimodal AI System for **Customer Churn Prediction** & **Sentiment Analysis**

Transforming Telecom CRM using Machine Learning, Deep Learning & Business Intelligence 🚀

**Short summary:** This repository contains a production-focused project that predicts customer churn (structured data) and analyzes customer feedback sentiment (unstructured text). It combines XGBoost-based ML for churn, a CNN+BiLSTM DL model for sentiment, SHAP explainability, a Streamlit UI, SQLite-based history, and automated business insights. A full project report (Telco_Churn_Feedback_Report.pdf) is included. 

> 📌 Turning Raw Customer Data into **Actionable Insights** using AI-Powered Churn Intelligence
> 📄 Includes full project report: Telco_Churn_Feedback_Report.pdf

---


https://github.com/user-attachments/assets/ca3a6dcb-c24c-43f0-b3cf-62f219e7c8e9


---


https://github.com/user-attachments/assets/744090c7-ae65-4e4f-a65d-681f89e58751


---

## 📖 Table of Contents

1.  [Project Overview](#-1-project-overview)
2.  [Problem Statement](#-2-problem-statement)
3.  [Objectives and Outcomes](#-3-objectives-and-outcomes)
4.  [Dataset Description & Feature Engineering](#-4-dataset-description--feature-engineering)
5.  [Key Features & Capabilities](#-5-key-features--capabilities)
6.  [Models & Performance](#-6-models--performance)
7.  [Business Impact & Recommendations](#-7-business-impact--recommendations)
8.  [Power BI Business Dashboard](#-8-power-bi-business-dashboard-(executive-analytics))
9.  [Project Architecture & Tech Stack](#-9-project-architecture--tech-stack)
10. [UI Screenshots](#️-10-ui-screenshots--streamlit-interface)
11. [Installation & Setup](#-11-installation--setup)
12. [Usage Examples](#-12-usage-examples)
13. [Conclusion](#-13-conclusion) 
14. [Future Scope & Deployment](#-14-future-scope--deployment)
15. [Author](#-15-author)
16. [License](#-16-license)

---

## 🧠 1. Project Overview

Telecom companies lose **millions of dollars every year due to customer churn**. Understanding *who is likely to leave* and **why** is crucial for customer retention. This project builds a **multimodal AI system** to address this challenge by combining structured and unstructured data analysis.

- **Structured Data Analysis:** Utilizes traditional Machine Learning (ML) models (e.g., XGBoost) on customer demographics, service usage, and billing information to predict the likelihood of churn.
- **Unstructured Data Analysis:** Employs Deep Learning (DL) models (e.g., CNN + BiLSTM + Attention) for Natural Language Processing (NLP) to analyze the sentiment of customer feedback text.
- **Multimodal Fusion:** The system is designed to integrate the outputs of both models to provide a more robust and explainable prediction of customer behavior.
- **Business Intelligence dashboards:**  Power BI
- **Explainability & Insights:**  SHAP + dashboards
  
The final outcome is a  **production-ready CRM intelligence system** that supports **data-driven retention strategies**.

---

## ❗ 2. Problem Statement

- Customers leave telecom services due to **billing confusion**, **poor support**, or **network issues**.
- Business teams **lack AI-driven tools** to analyze customer feedback + behavior together.
- Manual CRM analysis is **slow**, **biased**, and **reactive**, instead of **proactive**.

> **The core question addressed by this project is:** Can an AI system effectively detect early churn indicators AND provide a clear explanation of *why* customers are leaving by synthesizing insights from both structured service data and unstructured feedback text?

This project demonstrates that a **combined ML + DL system** can successfully achieve this goal.

---

## 🎯 3. Objectives and Outcomes

| Goal | Outcome | Technology |
|:---|:---|:---|
| **Predict Churn** | High-accuracy churn prediction model | XGBoost Classifier |
| **Analyze Sentiment** | Deep Learning model for sentiment classification | CNN + BiLSTM + Attention |
| **Multimodal Learning** | Fusion of structured and text features for improved prediction | Feature Engineering |
| **Business Insights** | Early identification of at-risk customers and root causes | Model Explainability, Power BI |
| **Deployment Readiness** | Modular, production-ready code structure | Python, `app.py, Streamlit |

---

## 📂 4. Dataset Description & Feature Engineering

The project uses a **Telco Customer Churn dataset** enhanced with **Realistic Customer Feedback** for multimodal modeling.

**Source:** Telco Customer Churn with Realistic Customer Feedback
https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback

| Feature | Description | Data Type |
|:---|:---|:---|
| **Structured Data** | Tenure, Services (Phone, Internet, etc.), Billing, Charges | Numerical/Categorical |
| **Text Data** | `CustomerFeedback` | Unstructured Text (NLP) |
| **Target (ML)** | `Churn` (Yes/No) | Binary Classification |
| **Target (DL)** | `Derived Sentiment` (Positive/Neutral/Negative) | Multi-class Classification |

- **Total Records:** 7,043
- **Total Features:** 23
- **Key Insight:** The inclusion of both `Churn` and `Sentiment` labels enables the development of a powerful **multimodal AI model**.

---

### 🔧 Feature Engineering Added
The dataset was enhanced with features derived from the text data and business logic:

| New Feature | Purpose |
|:---|:---|
| `feedback_length`, `word_count` | Measures text complexity for NLP. |
| `sentiment_pos`, `sentiment_neg`, `sentiment_neu`, `sentiment_compound` | Sentiment scores (VADER) for fusion with ML model. |
| `is_high_value_customer` | VIP customer classification based on business rules. |
| `is_new_customer` | Flag for new/existing customer based on tenure. |

---

### 🛠 Final Model Features (Used in ML Model)
```python
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "feedback_length", "word_count",
    "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound",
    "is_high_value_customer", "is_new_customer",
]
```

**📌 Target Variables**
Target	Used For
Churn	Customer retention prediction
Sentiment_Label	Customer satisfaction understanding

**📥 Example Input** (CSV Format)
tenure,MonthlyCharges,TotalCharges,is_high_value_customer,is_new_customer,CustomerFeedback
36,70.5,2500,1,0,"Billing is confusing, service is slow."
12,45.0,540,0,1,"Service is good so far."
65,85.0,2580,1,0,"Internet speed terrible!"

---

## 🚀 5. Key Features & Capabilities

The application is built for deployment and includes a comprehensive set of features for real-world CRM use cases, primarily leveraging a **Streamlit** interface.

| Feature Category | Description | Business Value |
|:---|:---|:---|
| **Single Prediction** | Real-time churn probability and sentiment analysis for one customer. Supports "What-if" scenario testing. | **Proactive Intervention** by Customer Retention Assistant. |
| **Batch Prediction** | Upload multiple customer records (CSV) to get churn + sentiment scores. Results stored in **SQLite DB**. | **Scalable Analysis** for large customer segments. |
| **Business Dashboard** | Churn risk segmentation, KPI metrics, and AI-powered **Retention Strategy Recommendation** (AI Strategy Generator). | **Strategic Decision Making** and CLV estimation. |
| **Explainability (SHAP)** | Shows **Why the model predicted churn** (Local & Global SHAP plots). | **Trust and Transparency** in AI predictions for managers. |
| **Role-Based Access** | Unique feature with different dashboard access for Customer, CRM Manager, and Data Scientist roles. | **Secure and Targeted** access control. |
| **Automated Report** | Generates a **professional business summary PDF/CSV** containing metrics, insights, and recommendations. | **Easy Reporting** for executive stakeholders. |
| **History & Analytics** | View session & database history; filter **Positive / Negative / High-risk** customers. | **Continuous Monitoring** and historical trend analysis. |

---

## 📈 6. Models & Performance

The models were rigorously trained and evaluated to ensure robustness and prevent overfitting. The system uses a dual-model approach for multimodal analysis, with models selected for high performance.

### 🔹 Machine Learning – Churn Prediction (Structured Data)
| Model | Task | Metric | Score | Final Verdict |
|:---|:---|:---|:---|:---|
| **Logistic Regression** | Binary Classification (Churn: Yes/No) | **ROC-AUC** | **94.2%** | ❌ Rejected |
| **Random Forest** | Binary Classification (Churn: Yes/No) | **ROC-AUC** | **95.2%** | ❌ Rejected |
| **XGBoost Classifier** | Binary Classification (Churn: Yes/No) | **ROC-AUC** | **96.77%** | ⭐ Selected - for high predictive power, handles imbalance & feature importance |

### 🔹 Deep Learning – Sentiment Analysis (Unstructured Data)
| Model | Task | Architecture | Metric | Score | Final Verdict |
|:---|:---|:---|:---|:---|:---|
| **Deep Learning Model** | Multi-class Classification (Sentiment) | **CNN + BiLSTM + Attention** | **Accuracy** | **88.15%** | Positive Results for automated feedback analysis |

**DL Model Architecture:**
| Layer | Purpose |
|:---|:---|
| Embedding Layer | Word representation |
| Conv1D | Feature extraction |
| BiLSTM | Context understanding |
| Attention| Focus on important features |
| Dense Layer | Final classification (Positive / Neutral / Negative) |

The best models were automatically selected using a weighted scoring system, focusing on balanced performance metrics to ensure real-world applicability.

---

## 💡 7. Business Impact & Recommendations

The insights derived from this multimodal system translate directly into actionable business strategies, ensuring the AI solution drives measurable ROI.

| Insight | Business Action | Strategic Goal |
|:---|:---|:---|
| **Low Tenure = High Churn Risk** | Implement an enhanced, personalized onboarding program for new customers. | **Improve Retention** |
| **Negative Feedback Predicts Churn** | Automatically trigger an alert to the dedicated support team for immediate intervention. | **Proactive Service Recovery** |
| **High Billing Dissatisfaction** | Offer targeted retention plans or transparent billing explanations to affected segments. | **Reduce Dissatisfaction** |
| **Service-Specific Churn Drivers** | Prioritize investment and quality improvements in the services most frequently mentioned in negative feedback. | **Optimize Service Quality** |

---

## 📊 8. Power BI Business Dashboard (Executive Analytics)

### 🔍 Overview

In addition to the AI-driven Streamlit application, a Power BI dashboard was developed to provide executive-level churn, revenue, and sentiment insights using the cleaned and feature-engineered dataset.

This dashboard is designed for:

- Business Managers
- CRM Teams
- Non-technical Stakeholders

### 📈 Dashboard Pages

The Power BI report includes **four structured pages**:

| Page | Purpose |
|:---|:---|
| **Executive Overview** | High-level churn %, revenue impact, contract risk |
| **Customer Segmentation & Behavior** | Churn by tenure, service type, payment method |
| **Revenue & Value Impact** | Revenue concentration, high-value customers, churn loss | 
| **Sentiment & Feedback Insights** | Churn vs sentiment, customer feedback validation |

### 📌 Key Metrics Tracked

- Overall Churn Rate
- Revenue at Risk
- High-Value Customer Share
- Churn by Contract & Tenure
- Sentiment-driven churn risk

### 🧠 Business Value

- Converts ML outputs into decision-ready insights
- Enables interactive filtering using slicers
- Helps leadership prioritize retention strategies

### 📷 Screenshots:

**Page-1: Executive Overview**

<img width="1306" height="737" alt="Overview_Page_1" src="https://github.com/user-attachments/assets/33aebc71-6fea-4528-9f3a-2a58b48033fb" />

**Page-2: Customer Segmentation & Behavior**

<img width="1302" height="737" alt="Customers_page_2" src="https://github.com/user-attachments/assets/4a17f4b9-02b0-4801-9f7d-ef07d7513530" />

**Page-3: Revenue & Value Impact**

<img width="1301" height="735" alt="Revenue_page_3" src="https://github.com/user-attachments/assets/2ce7af9d-6a4f-4de9-99b9-cb334ebe3676" />

**Page-4: Sentiment & Feedback Insights**

<img width="1302" height="733" alt="Sentiment_page_4" src="https://github.com/user-attachments/assets/f79c71e7-374e-48e2-b856-87a7505e58fe" />

### 📁 **Location:**  
`/reports/Telco_Churn_Sentiment_Analysis.pbix`

---

## 🏗 9. Project Architecture & Tech Stack

The project follows a modular structure ready for production deployment.

### ⚙ Project Workflow

🔍 User inputs Structured + Feedback Text ➜ 📌 Feature Engineering (VADER + TF-IDF + NLP) ➜ 🤖 ML Model → Churn Prediction (XGBoost) ➜ 🧠 DL Model → Sentiment Analysis (CNN-BiLSTM) ➜ 💾 SQLite DB → Save History (Single/Batch) ➜ 📊 Dashboards + SHAP Explainability
   AI Business Insights + PDF Reports            
    
**🧠 Streamlit App Architecture:**
UI Pages (Streamlit):
- Single Prediction (Customer-level)
- Batch Prediction (CSV)
- Business Dashboard (KPIs & charts)
- Explainability (SHAP local & global)
- History & Export (SQLite backed)
- Admin / Model Info (for Data Scientist role)

**🔁 System-Level:**
 
 View USER ➜ STREAMLIT FRONTEND ➜ AI ENGINE ➜ PREDICTION 
                       
                        │   
                        ▼ 
                
                 SQLITE DATABASE 
                        │ 
                        ▼ 
              DASHBOARDS + REPORTS 
  
**⚙ Architecture Summary:**
  
  Layer What Happens 
 - 🎛 UI Layer Streamlit-based user interface
 - 🔍 ML Layer XGBoost churn prediction
 - 🧠 DL Layer CNN-BiLSTM sentiment analysis
 - 🧠 NLP Layer TF-IDF + VADER feature engineering
 - 🔎 SHAP Layer Model explainability & trust
 - 💾 DB Layer SQLite history storage
 - 📊 Viz Layer Dashboards & insights

---

### 🛠 Tech Stack — Technologies Used

| Category | Tools Used |
|:---|:---|
| **ML** | XGBoost, RandomForest, scikit-learn |
| **DL**| TensorFlow / Keras — CNN + BiLSTM (+ Attention) |
| **NLP** | TF-IDF, Vader |
| **Explainability** | SHAP |
| **Database** | SQLite (Persistent storage for history) |
| **Frontend** | Streamlit |
| **BI & Visualization** | Matplotlib, seaborn and Power BI |
| **Data Handling** | Pandas, NumPy |
| **Optional Deployment** | FastAPI, Streamlit web, github |

---

### 📁 Project Structure

```bash
/telecom-churn-multimodal-ai
├── data/                       # Raw, processed, and featured datasets
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and feature-engineered data
│   └── featured/               # Final feature sets
├── notebooks/                  # Jupyter Notebook for EDA and experimentation
│   └── telco_churn_analysis.ipynb
├── src/                        # Modular Python scripts (app.py, config.py, etc.)
│   ├── data_preprocessing.py   # Data cleaning and preparation
│   ├── feature_engineering.py  # Feature creation for ML/DL
│   ├── model_preparation.py    # Model definition and loading
│   └── model_training.py       # Training and evaluation pipeline
├── models/                     # Saved trained models (e.g., .pkl, .h5)
├── outputs/                    # Model evaluation reports and visualizations
├── reports/                    # Screenshots and generated reports
├── app.py                      # Main application file (Streamlit/FastAPI entry point)
├── config.py                   # Configuration settings and constants
├── utils.py
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation (You are here)
```

---

## 🖥️ 10. UI Screenshots — Streamlit Interface

The application features a **clean, interactive, and industry-ready Streamlit UI** designed for both **business users & data scientists**.

### 🔍 Single Customer Prediction Page 
> Predict churn probability & analyze sentiment from customer feedback.
>  | Structured Inputs (tenure, charges, flags) | NLP Sentiment Analysis (Text) |
>  |--------------------------------------------|-------------------------------|
>  | 🚀 Real-time prediction | 📊 AI-powered insights |
 
<img width="1920" height="2258" alt="churn_single_prediction_page" src="https://github.com/user-attachments/assets/61b9db5c-4270-4516-ad51-acb2189fb149" />

--- 
 
### 📂 Batch Prediction (CSV Upload) 
 > Upload CSV → get churn & sentiment prediction for hundreds of customers. 

<img width="1920" height="1893" alt="batch_prediction_page" src="https://github.com/user-attachments/assets/c0832797-19fc-4669-be33-79ee6dba8f8b" />

--- 
  
### 📊 Business Dashboard – Churn & Revenue Insights 
  > For CRM & management teams — full business analysis: - KPI Metrics - Revenue at Risk - Segmentation Analysis - Heatmaps & Churn Trends
   
   ![Uploading business_dashboard_page.png…]()

--- 
   
### 🧠 Explainability (SHAP) 
   > **Why did the model predict churn?** Explains prediction with **SHAP feature importance** — industry standard for trust & validation.
> | SHAP – Single Customer | SHAP – Batch Customer | SHAP – Global Feature Impact |
> |-----------------------|-----------------------|-------------------------------|
>  | Local reasoning | Multiple insights | Dataset-wide insights | 
   
**Single SHAP Tab**

<img width="1920" height="2645" alt="single_shap_page" src="https://github.com/user-attachments/assets/bc04fba1-534d-4660-af1e-390c59f19e4d" />

**Batch SHAP Tab**
<img width="1920" height="4019" alt="batch_shap_page" src="https://github.com/user-attachments/assets/552f0ac5-4b87-4cb3-9045-7abc8631349d" />

**Global SHAP Tab**
<img width="1920" height="3523" alt="global_shap_page" src="https://github.com/user-attachments/assets/47f0f4a0-1935-4357-bb86-0540d6a2b2ff" />

--- 
   
### 📁 Insights & History Center 
   > Tracks **all previous predictions** using SQLite database & session storage. Supports **filter, download, restore & backup** options. 

**Single History Tab**
<img width="1920" height="3533" alt="single_history_page" src="https://github.com/user-attachments/assets/35a21176-48e4-4df5-82c2-19e90c9dffb7" />

**Batch History Tab**
<img width="1920" height="3167" alt="Batch_history_page" src="https://github.com/user-attachments/assets/5f56d699-a4b0-4b8d-b07e-0ec7348da56b" />

**Summary Insights Tab**
<img width="1920" height="5244" alt="summary_insights_page" src="https://github.com/user-attachments/assets/3ad0dd70-edc0-4f2b-be0c-91a445742a15" />

**BackUp and Restore Tab**
<img width="1920" height="2158" alt="history_insights_page" src="https://github.com/user-attachments/assets/126caa60-89e8-41c7-b5da-bef2afe60920" />

--- 

### ! Help & Info 

<img width="1920" height="4132" alt="help_info_page" src="https://github.com/user-attachments/assets/ba87441b-f45a-4c96-a39c-2fa4cdf18136" />

---

### 🧠 AI Suggestions & Retention Strategy
> Auto-generated **business recommendations** based on churn + sentiment ✨ 
    
--- 
    
### 🧭 Role-Based Access Control (RBAC) 
    > Different UI for 
    | Role | Purpose | 
    |------|--------|
    | 👤 Customer | Single prediction |
    | 👨‍💼 Business Manager | Batch + dashboard | 
    | 🧠 Data Scientist | SHAP + analytics |    

---

## ⚙ 11. Installation & Setup

1.  **Clone Repository**
    ```bash
    git clone https://github.com/Ayesha24banu/telecom_churn_multimodal_ai.git
    cd telecom_churn_multimodal_ai
    ```

2.  **Create & Activate Virtual Environment**
    ```bash
    conda create -n churn-env python=3.10
    conda activate churn-env
    # OR using venv:
    # python -m venv churn-env
    # source churn-env/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Streamlit App**
    ```bash
    streamlit run src/app.py
    ```

---

## 📌 12. Usage Examples

**Single prediction (UI)**
- Open Streamlit UI.
- Choose Single Prediction.
- Enter the customer's structured fields and feedback text.
- Submit → view churn probability, sentiment label, SHAP explanation and suggested retention actions.

**Batch prediction**
- Upload CSV (must include headers).
- App returns a downloadable CSV with churn_probability, churn_label, sentiment_label, and interpretation_notes.

**Exporting a Business Report**
From Dashboard → Generate Report → download PDF (includes KPIs, top churn drivers, sample SHAP explanations and recommended actions).

---

## 🧾 13. Conclusion
 This project demonstrates how **AI can transform Telecom CRM systems** by combining **customer behavior (structured data)** and **feedback sentiment (unstructured text)** to **predict churn, detect dissatisfaction & suggest business actions** — just like real industry systems.

- ✔ Predicts **who is at risk** 
- ✔ Understands **why they may churn** 
- ✔ Suggests **data-driven retention strategies** 
- ✔ Streamlit UI + Database + Explainability = **Industry-ready portfolio project** 
  
  > 💡 **This project is not just academic — It can be deployed as a real CRM tool.** > Shows strong skills in **ML + DL + NLP + Business Analytics**, perfect for **Interviews & Job Applications.** 

---

## 🔮 14. Future Scope & Deployment

The project is designed with future industry deployment in mind.

-   **API Deployment:** Transition the model serving layer to **FastAPI** or **Flask** for high-performance, scalable API access.
-   **Cloud Hosting:** Deploy the application on platforms like **AWS**, **Render**, or **Railway**.
-   **Authentication:** Integrate robust authentication using **Firebase** or **OAuth**.
-   **Chatbot Integration:** Develop a **Customer Support AI Bot** leveraging the sentiment analysis model.
-   **Automated Alerts:** Implement a system to automatically flag and alert the customer support team when a customer is predicted to be at high risk of churning.
-   **Real-time Monitoring:** Connect to a **Power BI Dashboard** for real-time monitoring of churn risk and sentiment trends.

---

## 👩‍💻 15. Author 

**👤 Author:** Ayesha Banu (Gold Medalist – MSc Computer Science)

**🔍 Role:** Aspiring  Data Analyst | ML Engineer | Data Scientist 

**Contact:**
-   📧 Email: ayesha24banu@gmail.com
-   🔗 LinkedIn: https://www.linkedin.com/in/ayesha-banu-cs/
-   ⭐ GitHub: https://github.com/Ayesha24banu

> “Turning Data into Decisions & AI into Business Value.”

---

### 🏁 Final Remark 

> “This project helped me combine Machine Learning + Deep Learning + Business Thinking to solve real customer problems. I am excited to apply these skills to industry projects and full-time roles in Data Science / ML Engineering / AI Development.” 
- 🙏 Thank you for viewing this project! 
- ⭐ If you found it useful, don’t forget to star ⭐ the repository! 
- 🚀 Open for feedback, contributions & collaborations.

---

### 📄 References

 - Telco Customer Churn Dataset. *Kaggle*. [https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback]
 - Deep Learning for Sentiment Analysis. *Journal of Artificial Intelligence Research*. [https://www.jair.org/index.php/jair/article/view/11364]
 - XGBoost Documentation
 - SHAP Explainability Framework

**Key libs:** XGBoost, scikit-learn, TensorFlow/Keras, SHAP, Streamlit.

> 📌 **Key Research Papers & Blogs:** 
> - "Customer Churn Prediction in Telecom using ML" – IEEE Research 
> - "Sentiment Analysis using Neural Networks" 

---

## 📜 16. License

This project is released under the MIT License — feel free to use for learning, portfolios, or production with attribution.

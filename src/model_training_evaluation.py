# MODELS BUILDING, TRAINING AND EVALUATION (ML + DL)
import os
import time
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# DL MODELS (Keras / TensorFlow)
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D,
    MaxPooling1D, Bidirectional, LSTM, Input, Flatten, Add
)

from tensorflow.keras.callbacks import CSVLogger
from config import OUTPUTS_DIR


# Helper: EVALUATE MODEL PERFORMANCE (ML & DL)
def evaluate_model(model, X_train, y_train, X_test, y_test, keras=False):
    """
    Evaluates model performance:
    - keras=False  → ML model (binary classification)
    - keras=True   → DL model (multi-class)
    Returns accuracy, F1, ROC-AUC (if applicable), confusion matrix
    """
    try:
        if keras: #  DEEP LEARNING → Softmax Output
            y_train_prob = model.predict(X_train)
            y_test_prob  = model.predict(X_test)

            # Convert softmax probabilities to actual labels
            y_train_pred = np.argmax(y_train_prob, axis=1)
            y_test_pred  = np.argmax(y_test_prob, axis=1)

        else:  #  ML MODELS (BINARY CHURN)
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)
            
            try:
                y_test_prob = model.predict_proba(X_test)[:, 1]  # if available
            except:
                y_test_prob = y_test_pred  # Fallback if no predict_proba()

        metrics = {
            "train_acc": accuracy_score(y_train, y_train_pred),
            "test_acc": accuracy_score(y_test, y_test_pred),
            "train_f1": f1_score(y_train, y_train_pred, average='weighted'),
            "test_f1": f1_score(y_test, y_test_pred, average='weighted'),
            "classification_report": classification_report(y_test, y_test_pred),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
        }

        # Only ML binary models should compute ROC-AUC
        try:
            if not keras and len(np.unique(y_test)) == 2:
                metrics["test_roc_auc"] = roc_auc_score(y_test, y_test_prob)
            else:
                metrics["test_roc_auc"] = None
        except:
            metrics["test_roc_auc"] = None

        return metrics

    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise


# Helper: CHECK OVERFITTING / UNDERFITTING / DATA LEAKAGE
def check_model_fit(train_acc, test_acc, train_f1, test_f1):
    """
    Detects:
    ❗ Overfitting  → High train but low test accuracy
    ❗ Underfitting → Both train & test low
    ❗ TOO PERFECT  → Possible leakage ⚠
    """
    remark = "✔ Good Fit"
    overfit = False
    underfit = False

    if abs(train_acc - test_acc) > 0.10:
        remark = "⚠ Overfitting"
        overfit = True

    if train_acc < 0.65 and test_acc < 0.65:
        remark = "⚠ Underfitting"
        underfit = True

    if train_acc > 0.98 and test_acc > 0.98:
        remark = "⚠ TOO PERFECT — Possible TF-IDF Leakage"

    return {"overfit": overfit, "underfit": underfit, "remark": remark}

    
# Helper: WEIGHTED MODEL SCORING FOR BEST MODEL SELECTION
def compute_weighted_score(test_acc, test_roc_auc, test_f1, train_acc, train_f1):
    """
    Final ML & DL score for model selection:
    50% Test Accuracy
    30% ROC-AUC
    20% F1 Score
    -10% Penalty if overfitting
    """
    # fallback for missing roc_auc
    roc = test_roc_auc if (test_roc_auc is not None) else (0.5 * test_f1)
    base = 0.50 * test_acc + 0.30 * roc + 0.20 * test_f1

    # apply penalty if overfitting
    flags = check_model_fit(train_acc, test_acc, train_f1, test_f1)
    penalty = 0.10 if flags["overfit"] else 0.0

    return base - penalty


# MAIN TRAINING FUNCTION (ML TRAINING AND EVALUATE — ONLY STRUCTURED)
def train_ml_models(X_train_ml, X_test_ml, y_train_ml, y_test_ml):
    '''
    Trains Logistic Regression, Random Forest, XGBoost
    Record time, GridSearchCV used
    Over/Underfit checks warning
    Select best model automatically using Weighted score
    '''

    print("\n===== TRAINING ML MODELS (Churn Prediction) =====")
    logging.info("Training started")
    
    ml_results = {}
    best_ml_model = None
    best_ml_name = None
    best_ml_score = -1

    # Convert sparse matrix (csr) to array if needed for ML only
    X_train_ml_arr = X_train_ml.toarray() if hasattr(X_train_ml, "toarray") else np.array(X_train_ml)
    X_test_ml_arr  = X_test_ml.toarray()  if hasattr(X_test_ml, "toarray") else np.array(X_test_ml)

    # ML models to run in a loop (GridSearchCV) — ONLY CHURN
    start_time = time.time()
    # Define models & parameters and basic hyperparameter grid
    ml_models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state= 42),
            "params": {"C": [0.1, 1, 10]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "params": {"max_depth": [None, 10], "n_estimators": [150, 300]}
        },
        # We compute scale_pos_weight using y_train to help XGBoost for imbalance
        "XGBoost": {
            "model": xgb.XGBClassifier( eval_metric="logloss", n_estimators=200, random_state=42,
            scale_pos_weight=((sum(y_train_ml == 0) / sum(y_train_ml == 1)))),
            "params": {"max_depth": [3, 5], "learning_rate": [0.1, 0.05]}
        }
    }

    # ML model Training + Evaluation
    for name, item in ml_models.items():
        try:
            print(f"\nTraining {name} with GridSearchCV...\n")
            logging.info(f"\nTraining {name} with GridSearchCV...\n")
            model_start = time.time()

            model = item["model"]
            param_grid = item["params"]

            # Simple GridSearchCV
            grid = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_train_ml_arr, y_train_ml)
            model_end = time.time()
            best_estimator = grid.best_estimator_
        
            model_time = model_end - model_start
            print(f"{name} training completed in {model_time:.2f} seconds")

            # Evaluate
            metrics = evaluate_model(best_estimator, X_train_ml_arr,  y_train_ml, X_test_ml_arr, y_test_ml)
            train_acc = metrics["train_acc"]
            test_acc = metrics["test_acc"]
            train_f1 = metrics["train_f1"]
            test_f1 = metrics["test_f1"]
            roc_auc = metrics["test_roc_auc"]

            # Print summary
            print("Best Params:", grid.best_params_)
            print("Train Accuracy: %.4f | Test Accuracy: %.4f | Test ROC-AUC: %s" % (train_acc, test_acc, str(roc_auc)))
            print("classification_report:", metrics["classification_report"])
            print("Confusion Matrix:", metrics["confusion_matrix"] )

            # Over/Underfit checks warning 
            fit_check = check_model_fit(train_acc, test_acc, train_f1, test_f1)
            print("Fit Status:", fit_check["remark"])

            # Weighted score for model selection
            final_score = compute_weighted_score(test_acc, roc_auc, test_f1, train_acc, train_f1)

            # Save results
            ml_results[name] = {
                "best_params": grid.best_params_,
                "time_sec": round(model_time, 3),
                "metrics": metrics,
                "remark": fit_check["remark"],
                "final_score": round(final_score, 4)
            }
            
            logging.info("%s finished. Time: %.2fs Score: %.4f", name, model_time, final_score)

            # Update best ML model by final_score
            if final_score > best_ml_score:
                best_ml_score = final_score
                best_ml_model = best_estimator
                best_ml_name = name

        except Exception as e:
            logging.error("Error training %s : %s", name, e)
            print(f"Error training {name}: {e}")

    # All models Total Time, Best ML model and return ml results
    total_time = time.time() - start_time
    print(f"\n BEST ML MODEL → {best_ml_name}  (Score: {best_ml_score:.4f})")
    print(f"Total training time: {total_time:.2f} seconds")
    logging.info("ML loop done. Best ML model: %s (score=%.4f). Total training time: %.2fs", best_ml_name, best_ml_score if best_ml_score is not None else -1, total_time)
    return {
        "ml_results": ml_results,
        "best_ml_model": best_ml_model,
        "best_ml_name": best_ml_name,
        "best_ml_score": best_ml_score
    }

    
# MAIN TRAINING FUNCTION (DL TRAINING AND EVALUATE — ONLY SENTIMENT (3 classes))
def train_dl_model(X_train_dl, X_test_dl, y_train_dl, y_test_dl):
    '''
    Sentiment Analysis (3 classes), Record time
    Handles class imbalance
    CNN + BiLSTM + MultiHeadAttention
    Over/Underfit checks warning
    Select best model automatically using Weighted score
    '''
    try:
        print("\n===== TRAINING DL MODEL (Sentiment Classification) =====")
        logging.info("Start training DL model")
        
        dl_results = {}
        best_dl_model = None
        best_dl_score = -1
        
        start = time.time()
        # Reshape for CNN/LSTM
        X_train_dl = X_train_dl.toarray().reshape(-1, X_train_dl.shape[1], 1)
        X_test_dl  = X_test_dl.toarray().reshape(-1, X_test_dl.shape[1], 1)
    
        # CLASS WEIGHTS (Handle Imbalance)
        classes = np.unique(y_train_dl)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_dl)
        class_weights = {cls: w for cls, w in zip(classes, weights)}

        #  BUILD DL MODEL (CNN + BiLSTM)
        inputs = Input(shape=(X_train_dl.shape[1], 1))
        # Reshape TF-IDF vectors → 3D for Conv1D & LSTM
        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x_res = Dense(128)(x)  # RESIDUAL CONNECTION (must match shape!)
        # LSTM block 
        x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        # MultiHeadAttention
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x) 
        # Residual connection 
        x = Add()([x, x_res])
        # Flatten & Dense layers
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(3, activation="softmax")(x)  # THREE Sentiment Classes

        # Compile
        dl_model = Model(inputs, outputs)
        dl_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # TRAIN DL MODEL
        # Early Stopping + ReduceLROnPlateau (LR Scheduler) added
        early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        # Save training history to CSV automatically inside OUTPUTS_DIR
        csv_log_path = os.path.join(OUTPUTS_DIR, "dl_training_log.csv")
        csv_logger = CSVLogger(csv_log_path, append=False)
        dl_model.fit(
            X_train_dl, y_train_dl,
            validation_split=0.1,
            epochs=20,
            batch_size=64,
            callbacks=[early_stop, reduce_lr, csv_logger],
            class_weight=class_weights,
            verbose=1
        )
        dl_time  = time.time() - start
        
        #  EVALUATE DL MODEL
        dl_metrics = evaluate_model(dl_model, X_train_dl, y_train_dl, X_test_dl, y_test_dl, keras=True)

        # Extract metrics
        train_acc = dl_metrics["train_acc"]
        test_acc = dl_metrics["test_acc"]
        train_f1 = dl_metrics["train_f1"]
        test_f1 = dl_metrics["test_f1"]
        roc_auc = dl_metrics["test_roc_auc"]

        # Over/Underfit checks warning 
        fit_check = check_model_fit(train_acc, test_acc, train_f1, test_f1) 
        # Weighted score for model selection
        dl_final_score = compute_weighted_score(test_acc, roc_auc, test_f1, train_acc, train_f1)

        # Save results
        dl_results["Neural Network (DL)"] = {
            "time_sec": round(dl_time, 3),
            "metrics": dl_metrics,
            "remark": fit_check["remark"],
            "final_score": round(dl_final_score , 4)
        }

        # Print summary
        print(f"Neural Network completed in {dl_time:.2f} sec")
        print("Test Accuracy: %.4f | TestF1 Score: %.4f" % (test_acc, test_f1))
        print("classification_report:", dl_metrics["classification_report"])
        print("Confusion Matrix:", dl_metrics["confusion_matrix"] )
        print("Fit Status:", fit_check["remark"]) 
        print(f"\n Best DL Model → Neural Network (Score: {dl_final_score:.4f})")
        dl_model= dl_model
        dl_score = dl_final_score
        logging.info("DL model finished. Neural Network, Time: %.2fs Score: %.4f", dl_time, dl_final_score )

    except Exception as e:
        logging.error("Error training DL model: %s", e)
        print("Error training neural network:", e)
        dl_model = None
        dl_score = -1

    # Return dl results
    return {
        "dl_results": dl_results,
        "best_dl_model": dl_model,
        "best_dl_score": dl_score
    }
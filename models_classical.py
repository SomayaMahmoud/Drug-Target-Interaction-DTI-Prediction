# models_classical.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTORS:
#   Person E — Ridge Regression (linear baseline with L2 regularisation)
#               Random Forest Regressor (100 trees, sqrt features)
#   Person D — Random Forest Classifier + GridSearchCV (Steps 8–13)
#               XGBoost Classifier (Step 13) + Regressor (Step 17)
#               Feature selection with SelectKBest (Step 20)
#               Hyperparameter tuning (Step 21)
#
# Person D used classification (predict binder/non-binder).
# Person E used regression (predict continuous pKd value).
# Both approaches are preserved here.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import Ridge
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost                 import XGBClassifier, XGBRegressor

from config import SEED


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ridge Regression  (Person E — Phase 2a)
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(X_train_scaled, y_train_reg):
    """
    Person E's baseline model.
    Ridge = Linear Regression + L2 penalty on weights.
    Loss = MSE + alpha × sum(w²)
    alpha=1.0 is a sensible default — larger = simpler model.

    MUST use scaled features (X_train_scaled, not raw X_train).
    Tree models don't care about scale; linear models do.
    """
    print("[classical] Training Ridge Regression (Person E)…")
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train_reg)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Forest Regressor  (Person E — Phase 2a)
# ─────────────────────────────────────────────────────────────────────────────

def train_rf_regressor(X_train, y_train_reg, sample_wts=None):
    """
    Person E's Random Forest for regression.
    100 trees, sqrt feature sampling, min 2 samples per leaf.
    Does NOT need scaled features.
    """
    print("[classical] Training Random Forest Regressor (Person E)…")
    model = RandomForestRegressor(
        n_estimators   = 100,
        max_features   = "sqrt",
        min_samples_leaf= 2,
        random_state   = SEED,
        n_jobs         = -1,
    )
    model.fit(X_train, y_train_reg,
              sample_weight=sample_wts if sample_wts is not None else None)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest Classifier + GridSearchCV  (Person D — Steps 8-13)
# ─────────────────────────────────────────────────────────────────────────────

def train_rf_gridsearch(X_train, y_train_cls):
    """
    Person D's approach:
        RandomForestClassifier → GridSearchCV over n_estimators and max_depth.
    Person D used:
        params = {"n_estimators": [100, 200], "max_depth": [10, 20]}
        GridSearchCV(RandomForestClassifier(), params, cv=3)

    Uses SMOTE-balanced training data (X_balanced, y_bal_cls).
    """
    print("[classical] Training RF + GridSearch (Person D)…")
    params = {
        "n_estimators": [100, 200],
        "max_depth"   : [10, 20],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        params, cv=3, n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train_cls)
    print(f"  Best params: {grid.best_params_}")
    return grid.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost Classifier  (Person D — Steps 13, 17, 21)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgb_classifier(X_train, y_train_cls):
    """
    Person D's XGBoost classifier.
    Person D's Step 21 tuned: n_estimators=300, max_depth=6, lr=0.1
    We use lr=0.05 (more conservative = better generalisation).

    Uses SMOTE-balanced training data.
    NOTE: verbose=0 goes in __init__ — not in .fit() (causes TypeError in XGBoost 2+).
    """
    print("[classical] Training XGBoost Classifier (Person D)…")
    model = XGBClassifier(
        n_estimators  = 300,
        max_depth     = 6,
        learning_rate = 0.05,
        subsample     = 0.8,
        colsample_bytree= 0.8,
        eval_metric   = "logloss",
        random_state  = SEED,
        n_jobs        = -1,
        verbosity     = 0,    # ← here, NOT in .fit()
    )
    model.fit(X_train, y_train_cls)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. XGBoost Regressor  (best combined approach)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgb_regressor(X_train, y_train_reg, sample_wts=None):
    """
    XGBoost for regression — predicts continuous pKd values.
    This is the best-performing classical model overall.
    """
    print("[classical] Training XGBoost Regressor…")
    model = XGBRegressor(
        n_estimators    = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = SEED,
        n_jobs          = -1,
        verbosity       = 0,   # ← here, NOT in .fit()
    )
    model.fit(X_train, y_train_reg,
              sample_weight=sample_wts if sample_wts is not None else None)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train all classical models and return predictions
# ─────────────────────────────────────────────────────────────────────────────

def train_all_classical(splits: dict) -> tuple:
    """
    Run every classical model.
    Returns (predictions_dict, models_dict).
    predictions_dict maps name → predicted pKd array on the test set.
    """
    X_tr     = splits["X_train"]
    X_te     = splits["X_test"]
    yr_tr    = splits["y_train_reg"]
    yr_te    = splits["y_test_reg"]
    yc_te    = splits["y_test_cls"]
    X_bal    = splits["X_balanced"]
    yc_bal   = splits["y_bal_cls"]
    X_sel_tr = splits["X_sel_train"]
    X_sel_te = splits["X_sel_test"]
    sw       = splits["sample_wts"]

    # Scaled X for Ridge
    fp_e  = splits["fp_end"]
    de_e  = splits["desc_end"]
    X_tr_sc = X_tr.copy()
    X_te_sc = X_te.copy()

    preds  = {}
    models = {}

    # ── Ridge (Person E) ────────────────────────────────────────────────────
    ridge = train_ridge(X_tr_sc, yr_tr)
    preds["Ridge (Person E)"]   = ridge.predict(X_te_sc)
    models["Ridge (Person E)"]  = ridge

    # ── RF Regressor (Person E) ──────────────────────────────────────────────
    rf_reg = train_rf_regressor(X_tr, yr_tr, sw)
    preds["RF Regressor (Person E)"]  = rf_reg.predict(X_te)
    models["RF Regressor (Person E)"] = rf_reg

    # ── RF + GridSearch (Person D) ───────────────────────────────────────────
    rf_gs = train_rf_gridsearch(X_bal, yc_bal)
    # convert classifier probabilities to pKd proxy
    proba = rf_gs.predict_proba(X_te)[:,1]
    yr_range = yr_tr.max() - yr_tr.min()
    preds["RF+GridSearch (Person D)"]  = proba * yr_range + yr_tr.min()
    models["RF+GridSearch (Person D)"] = rf_gs

    # ── XGBoost Classifier + SelectKBest (Person D) ──────────────────────────
    xgb_cls = train_xgb_classifier(X_sel_tr, yc_bal)
    proba2   = xgb_cls.predict_proba(X_sel_te)[:,1]
    preds["XGBoost Cls+KBest (D)"]  = proba2 * yr_range + yr_tr.min()
    models["XGBoost Cls+KBest (D)"] = xgb_cls

    # ── XGBoost Regressor (best model) ───────────────────────────────────────
    xgb_reg = train_xgb_regressor(X_tr, yr_tr, sw)
    preds["XGBoost Regressor"]  = xgb_reg.predict(X_te)
    models["XGBoost Regressor"] = xgb_reg

    return preds, models


# standalone test
if __name__ == "__main__":
    print("models_classical.py: import and call train_all_classical(splits)")

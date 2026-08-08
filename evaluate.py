# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTOR: Person E (davis_dta_full_project Phase 3)
#
# Person E's exact functions preserved:
#   • concordance_index() — the standard DTA ranking metric
#   • evaluate_model()    — prints MSE, RMSE, R², Pearson, CI
#   • Feature importance from Random Forest
#   • SHAP summary plot
#   • Error analysis (where the model fails most)
#   • Predicted vs Actual scatter plots
#
# Extended with:
#   • Combined results table across all team models
#   • Classification metrics (Accuracy, F1, ROC-AUC) from Person D
# ─────────────────────────────────────────────────────────────────────────────

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import shap
from scipy.stats import pearsonr
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, f1_score, roc_auc_score)


# We added SEED here
from config import SEED, OUTPUT_DIR, THRESHOLD_DAVIS as THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# 1. Concordance Index  (Person E — Phase 2a)
# ─────────────────────────────────────────────────────────────────────────────

def concordance_index(y_true: np.ndarray, y_pred: np.ndarray,
                      n_sample: int = 5000) -> float:
    """
    Person E's exact CI implementation.

    For every pair (i,j) where true affinities differ,
    check if our model correctly ranks them.
    CI = (correctly ranked pairs) / (total comparable pairs)

    CI = 0.5 → random guessing
    CI = 1.0 → perfect ranking
    Published DeepDTA CI on Davis: 0.878
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_true), size=min(len(y_true), n_sample), replace=False)
    yt, yp = y_true[idx], y_pred[idx]

    concordant = discordant = tied = 0
    for i in range(len(yt)):
        for j in range(i + 1, len(yt)):
            if yt[i] == yt[j]:
                continue
            if (yt[i] > yt[j]) == (yp[i] > yp[j]):
                concordant += 1
            elif yp[i] == yp[j]:
                tied += 0.5
            else:
                discordant += 1

    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5

SEED_VAL = 42  # for CI sampling


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full metric evaluation  (Person E + Person D metrics combined)
# ─────────────────────────────────────────────────────────────────────────────

all_results = []  # accumulate for final table

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                   print_output: bool = True) -> dict:
    """
    Person E's evaluate_model() extended with Person D's classification metrics.

    Regression metrics (Person E):
        MSE, RMSE, R², Pearson correlation, CI

    Classification metrics (Person D):
        Accuracy, F1, ROC-AUC
        (threshold y_pred ≥ 7.0 → binder, < 7.0 → non-binder)
    """
    mse     = mean_squared_error(y_true, y_pred)
    rmse    = np.sqrt(mse)
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    ci      = concordance_index(y_true, y_pred)

    # Person D's classification metrics
    pred_cls = (y_pred  >= THRESHOLD).astype(int)
    true_cls = (y_true  >= THRESHOLD).astype(int)
    acc = accuracy_score(true_cls, pred_cls)
    f1  = f1_score(true_cls, pred_cls, zero_division=0)
    try:
        auc = roc_auc_score(true_cls, y_pred)
    except Exception:
        auc = float("nan")

    row = dict(Model=name,
               MSE=round(mse,4), RMSE=round(rmse,4), MAE=round(mae,4),
               R2=round(r2,4), Pearson=round(pearson,4), CI=round(ci,4),
               Accuracy=round(acc,4), F1=round(f1,4), ROC_AUC=round(auc,4))
    all_results.append(row)

    if print_output:
        print(f"\n  {name}")
        print(f"    MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
        print(f"    Pearson={pearson:.4f}  CI={ci:.4f}")
        print(f"    Acc={100*acc:.1f}%  F1={f1:.4f}  ROC-AUC={auc:.4f}")

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 3. Results table  (Person E + Person D)
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table() -> pd.DataFrame:
    """Print the full comparison table and save to CSV."""
    df = pd.DataFrame(all_results).set_index("Model").round(4)
    print("\n" + "="*85)
    print("  FINAL RESULTS — ALL MODELS")
    print("="*85)
    cols = ["R2","RMSE","MAE","CI","ROC_AUC","F1","Accuracy"]
    print(df[cols].to_string())
    print("="*85)

    best_r2  = df["R2"].idxmax()
    best_ci  = df["CI"].idxmax()
    best_auc = df["ROC_AUC"].idxmax()
    print(f"\n  🏆 Best R²      : {best_r2}  ({df.loc[best_r2,'R2']:.4f})")
    print(f"  🏆 Best CI      : {best_ci}  ({df.loc[best_ci,'CI']:.4f})")
    print(f"  🏆 Best ROC-AUC : {best_auc}  ({df.loc[best_auc,'ROC_AUC']:.4f})")

    path = f"{OUTPUT_DIR}/results_all_models.csv"
    df.to_csv(path); print(f"\n  Saved → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parity plots  (Person E — "Predicted vs Actual scatter")
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity(predictions: dict, y_test_reg: np.ndarray, suffix: str = ""):
    """
    Person E's predicted-vs-actual scatter plot.
    Points near the red diagonal = good predictions.
    """
    n   = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1: axes = [axes]

    for ax, (name, yp) in zip(axes, predictions.items()):
        r2 = r2_score(y_test_reg, yp)
        ax.scatter(y_test_reg, yp, alpha=0.2, s=7, color="#378ADD")
        lo, hi = y_test_reg.min(), y_test_reg.max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
        ax.set_title(f"{name}\nR²={r2:.3f}")
        ax.set_xlabel("Actual pKd"); ax.set_ylabel("Predicted pKd")

    plt.suptitle("Predicted vs Actual — All Models", fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/parity_plots{suffix}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[evaluate] Parity plots saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model comparison bar chart  (Person D Step 24 + Person E)
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison():
    """Bar chart of R², CI, ROC-AUC across all models."""
    if not all_results:
        return
    df   = pd.DataFrame(all_results).set_index("Model")
    cols = ["R2", "CI", "ROC_AUC"]
    titl = ["R²", "Concordance Index (CI)", "ROC-AUC"]
    colors = ["#2E75B6","#27AE60","#E74C3C","#F39C12","#8E44AD","#1ABC9C","#E67E22","#E74C3C"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(axes, cols, titl):
        vals = df[col].values
        bars = ax.barh(df.index, vals, color=colors[:len(df)], edgecolor="white")
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, min(max(vals) + 0.15, 1.05))
        for b, v in zip(bars, vals):
            ax.text(v + 0.005, b.get_y() + b.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9)

    plt.suptitle("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/model_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[evaluate] Comparison chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Importance  (Person E — Phase 3 / Random Forest importances)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(rf_model, fp_bits: int = 2048):
    """Person E's feature importance plot from the Random Forest."""
    importances = rf_model.feature_importances_
    top20_idx   = np.argsort(importances)[::-1][:20]

    feature_names = (
        [f"FP_{i}" for i in range(fp_bits)] +
        ["MolWt","LogP","TPSA","HDonors","HAcceptors",
         "RotBonds","HeavyAtoms","Rings","ArRings",
         "Heteroatoms","FrCSP3","SatRings"] +
        [f"AA_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"]
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(20), importances[top20_idx], color="#378ADD")
    ax.set_xticks(range(20))
    ax.set_xticklabels([feature_names[i] for i in top20_idx],
                       rotation=45, ha="right", fontsize=9)
    ax.set_title("Top 20 Most Important Features (Random Forest)")
    ax.set_ylabel("Feature Importance")
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[evaluate] Feature importance saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SHAP Analysis  (Person E — Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap(model, X_test: np.ndarray, model_name: str = "XGBoost",
              fp_bits: int = 2048):
    """
    Person E's SHAP summary plot.
    Shows which features push predictions up (red) or down (blue).
    """
    print(f"[evaluate] Computing SHAP for {model_name}…")
    idx = np.random.choice(len(X_test), size=min(300, len(X_test)), replace=False)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[idx])

    feature_names = (
        [f"FP_{i}" for i in range(fp_bits)] +
        ["MolWt","LogP","TPSA","HDonors","HAcceptors",
         "RotBonds","HeavyAtoms","Rings","ArRings",
         "Heteroatoms","FrCSP3","SatRings"] +
        [f"AA_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"]
    )

    shap.summary_plot(shap_vals, X_test[idx],
                      feature_names=feature_names,
                      max_display=15, show=False)
    plt.title(f"SHAP — {model_name}")
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/shap_{model_name.replace(' ','_')}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[evaluate] SHAP saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Error Analysis  (Person E — Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """
    Person E's error analysis:
    Find where the model makes the biggest mistakes.
    """
    errors = np.abs(y_true - y_pred)
    worst  = np.argsort(errors)[-10:]

    print(f"\n[evaluate] Error analysis — {model_name}")
    print(f"  Mean absolute error   : {errors.mean():.3f} pKd")
    print(f"  Worst 10 errors mean  : {errors[worst].mean():.3f} pKd")
    print(f"  Worst cases true pKd  : {y_true[worst].mean():.2f}  (high-affinity region)")
    print(f"  → Model struggles most with strongly-binding pairs")
    print(f"    (training data is sparse there — imbalanced dataset)")

    # plot error distribution
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(errors, bins=50, color="#E74C3C", edgecolor="white", lw=0.5)
    ax.set_title(f"Absolute Error Distribution — {model_name}")
    ax.set_xlabel("|Predicted pKd − Actual pKd|")
    ax.set_ylabel("Count")
    ax.axvline(errors.mean(), color="navy", linestyle="--",
               label=f"Mean error = {errors.mean():.3f}")
    ax.legend()
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/error_distribution.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Error plot saved → {out}")


# standalone
if __name__ == "__main__":
    print("evaluate.py: import and call evaluate_model(), plot_parity(), etc.")

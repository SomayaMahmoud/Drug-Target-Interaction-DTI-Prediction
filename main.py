# main.py — runs the entire pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Just run:  python main.py
# or in Colab: paste all files + !python main.py
# ─────────────────────────────────────────────────────────────────────────────

import time, warnings
warnings.filterwarnings("ignore")

from data             import load_dataset, explore
from features         import build_features, split_scale_balance, build_tfidf_features
from models_classical import train_all_classical
from models_deep      import train_all_deep
from evaluate         import (evaluate_model, print_results_table,
                               plot_parity, plot_model_comparison,
                               plot_feature_importance, plot_shap, error_analysis)
from recommend        import recommend_drugs, plot_recommendations
from ui               import launch_ui
from config           import DATA_PATH_DAVIS

print("="*60)
print("  DTI PREDICTION PIPELINE — FULL TEAM")
print("="*60)

# ── Step 1: Data (Person D + E) ───────────────────────────────
print("\n── STEP 1: Load & Explore Data ──────────────────────────")
df = load_dataset(DATA_PATH_DAVIS, "davis")
explore(df, "davis")

# ── Step 2: Features (Person B + D + E) ──────────────────────
print("\n── STEP 2: Feature Engineering ──────────────────────────")
X, y_reg, y_cls, fp_end, desc_end = build_features(df)
splits = split_scale_balance(X, y_reg, y_cls, fp_end, desc_end)
X_tfidf, _, _ = build_tfidf_features(df)

# ── Step 3: Classical Models (Person D + E) ───────────────────
print("\n── STEP 3: Classical ML Models ──────────────────────────")
t0 = time.time()
cl_preds, cl_models = train_all_classical(splits)
print(f"Classical models done in {time.time()-t0:.0f}s")

# evaluate classical models
y_te = splits["y_test_reg"]
classical_parity = {}
for name, yp in cl_preds.items():
    evaluate_model(name, y_te, yp)
    classical_parity[name] = yp

# ── Step 4: Deep Learning (Person A + C + E) ──────────────────
print("\n── STEP 4: Deep Learning Models ────────────────────────")
t0 = time.time()
dl_preds, dl_models = train_all_deep(df, splits)
print(f"Deep learning done in {time.time()-t0:.0f}s")

for name, (yp, yte) in dl_preds.items():
    evaluate_model(name, yte, yp)

# ── Step 5: Results (Person D + E) ────────────────────────────
print("\n── STEP 5: Evaluation & Visualisation ──────────────────")
res_df = print_results_table()

# parity plots for classical models
plot_parity(classical_parity, y_te, suffix="_classical")
plot_model_comparison()

# feature importance (Person E)
if "RF Regressor (Person E)" in cl_models:
    plot_feature_importance(cl_models["RF Regressor (Person E)"])

# SHAP (Person E) — use best tree model
best_tree = cl_models.get("XGBoost Regressor") or cl_models.get("RF Regressor (Person E)")
if best_tree:
    plot_shap(best_tree, splits["X_test"], "XGBoost Regressor")

# error analysis (Person E)
best_name = res_df["R2"].idxmax()
if best_name in cl_preds:
    error_analysis(y_te, cl_preds[best_name], best_name)

# ── Step 6: Recommendation System ────────────────────────────
print("\n── STEP 6: Drug Recommendation System ──────────────────")
best_model  = cl_models.get("XGBoost Regressor") or list(cl_models.values())[0]
drug_library = df[["smiles"]].drop_duplicates()
sample_prot  = df["sequence"].iloc[0]

print(f"Recommending drugs for protein: {sample_prot[:60]}…")
top10 = recommend_drugs(
    target_sequence = sample_prot,
    drug_library_df = drug_library,
    model           = best_model,
    scaler          = splits["scaler"],
    fp_end          = fp_end,
    desc_end        = desc_end,
    top_k           = 10,
)
print("\nTOP 10 RECOMMENDED DRUGS:")
print(top10.to_string())
plot_recommendations(top10)

# ── Step 7: Gradio UI ─────────────────────────────────────────
print("\n── STEP 7: Launch Gradio UI ─────────────────────────────")
launch_ui(best_model, splits["scaler"], drug_library, fp_end, desc_end)

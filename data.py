# data.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTORS:
#   Person D — original data loading, cleaning, null removal,
#               binary label creation, KIBA + Davis handling
#   Person E — comprehensive EDA, affinity distribution plots,
#               protein sequence analysis, data insights
#
# Original issues fixed:
#   • Column names differ between Davis/KIBA — auto-renamed here
#   • Person D applied scaler.fit_transform(X) on everything including
#     binary fingerprints — fixed in features.py (scale only continuous cols)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")   # suppress RDKit deprecation warnings

from config import (DATA_PATH_DAVIS, DATA_PATH_KIBA,
                    OUTPUT_DIR, THRESHOLD_DAVIS, THRESHOLD_KIBA, SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN  (Person D — Steps 1-4 of DTI_Project.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str, dataset_name: str = "davis") -> pd.DataFrame:
    """
    Load a DTI CSV (Davis or KIBA), clean it, and return a standardised
    DataFrame with three columns: smiles | sequence | affinity

    Person D originally used:
        df = pd.read_csv("kiba_all.csv")
        df = df.dropna()
        df = df.drop_duplicates()
        df['label'] = df['affinity'].apply(lambda x: 1 if x > threshold else 0)

    Improvements here:
        • auto-rename columns so code works on both Davis and KIBA
        • validate SMILES with RDKit (remove unparseable rows)
        • threshold is dataset-aware
    """
    threshold = THRESHOLD_DAVIS if dataset_name.lower() == "davis" else THRESHOLD_KIBA

    df = pd.read_csv(path)
    print(f"[data] Loaded {len(df):,} rows from '{path}'")
    print(f"[data] Columns: {df.columns.tolist()}")

    # ── auto-rename to standard names ──────────────────────────────────────
    # Davis uses 'compound_iso_smiles', KIBA might differ — handle both
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "smiles"   in low: rename[col] = "smiles"
        if "sequence" in low: rename[col] = "sequence"
        if "affinity" in low: rename[col] = "affinity"
    df = df.rename(columns=rename)[["smiles", "sequence", "affinity"]].dropna()

    # ── remove duplicate pairs (Person D: drop_duplicates) ─────────────────
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[data] Duplicates removed: {before - len(df)}")

    # ── validate SMILES (Person D had None-check inside loops) ─────────────
    valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    removed    = (~valid_mask).sum()
    df = df[valid_mask].reset_index(drop=True)
    if removed:
        print(f"[data] Invalid SMILES removed: {removed}")

    # ── binary label (Person D: threshold-based classification) ────────────
    df["label"] = (df["affinity"] >= threshold).astype(int)

    print(f"[data] Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS  (Person E — Phase 1 of davis_dta_full_project)
# ─────────────────────────────────────────────────────────────────────────────

def explore(df: pd.DataFrame, dataset_name: str = "davis"):
    """
    Print dataset statistics and save EDA plots.

    Person E wrote detailed explanations for each chart.
    This function reproduces those charts as saved PNG files.
    """
    threshold = THRESHOLD_DAVIS if dataset_name.lower() == "davis" else THRESHOLD_KIBA
    binders    = (df["affinity"] >= threshold).sum()
    nonbinders = len(df) - binders
    imbalance  = nonbinders / max(binders, 1)

    # ── print summary (Person E's format) ──────────────────────────────────
    print("\n" + "="*58)
    print(f"  DATASET SUMMARY — {dataset_name.upper()}")
    print("="*58)
    print(f"  Total pairs      : {len(df):,}")
    print(f"  Unique drugs     : {df['smiles'].nunique():,}")
    print(f"  Unique proteins  : {df['sequence'].nunique():,}")
    print(f"  Affinity min     : {df['affinity'].min():.2f} pKd")
    print(f"  Affinity max     : {df['affinity'].max():.2f} pKd")
    print(f"  Affinity mean    : {df['affinity'].mean():.3f} pKd")
    print(f"  Affinity std     : {df['affinity'].std():.3f} pKd")
    print(f"  Binders (≥{threshold}) : {binders:,}  ({100*binders/len(df):.1f}%)")
    print(f"  Non-binders      : {nonbinders:,}  ({100*nonbinders/len(df):.1f}%)")
    print(f"  Imbalance ratio  : {imbalance:.1f}:1")
    print("="*58)

    pct5 = (df["affinity"] == 5.0).mean() * 100
    if pct5 > 50:
        print(f"\n⚠  {pct5:.1f}% of samples have affinity = 5.0 (floor value).")
        print("   Most drug-target pairs DO NOT bind.")
        print("   Accuracy alone is misleading — use ROC-AUC and F1.\n")

    # ── EDA plots (Person E's three-panel figure) ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: affinity histogram (Person E)
    axes[0].hist(df["affinity"], bins=50, color="#378ADD", edgecolor="white", lw=0.5)
    axes[0].axvline(df["affinity"].mean(), color="red", linestyle="--",
                    label=f"Mean: {df['affinity'].mean():.2f}")
    axes[0].axvline(threshold, color="orange", linestyle=":",
                    label=f"Threshold: {threshold}")
    axes[0].set_title("Affinity Distribution"); axes[0].set_xlabel("pKd")
    axes[0].legend(fontsize=8)

    # Plot 2: class imbalance bar (Person E + Person D)
    bins_aff = df["affinity"].apply(int).value_counts().sort_index()
    axes[1].bar(bins_aff.index, bins_aff.values,
                color=["#E24B4A" if i == 5 else "#378ADD" for i in bins_aff.index])
    axes[1].set_title("Class Imbalance"); axes[1].set_xlabel("Affinity (int bin)")

    # Plot 3: protein sequence length (Person E)
    seq_lens = df["sequence"].str.len()
    axes[2].hist(seq_lens, bins=40, color="#7F77DD", edgecolor="white", lw=0.5)
    axes[2].axvline(1000, color="red", linestyle="--", label="1000 aa cutoff")
    axes[2].set_title("Protein Sequence Lengths"); axes[2].set_xlabel("Length (aa)")
    axes[2].legend(fontsize=8)

    plt.suptitle(f"EDA — {dataset_name.upper()} Dataset", fontweight="bold", y=1.01)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/eda_{dataset_name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[data] EDA plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset(DATA_PATH_DAVIS, "davis")
    explore(df, "davis")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

# recommend.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTOR: All team (new addition built on everyone's work)
#
# Drug Recommendation System:
#   Given a target protein → score every drug in the library →
#   rank by predicted pKd → return Top-K binders
#
# This turns DTI prediction into a recommendation problem:
#   "Which drugs are most likely to bind THIS protein?"
#
# Technically it is content-based filtering:
#   The model learned drug-protein affinity patterns (content features).
#   We query it for a specific protein against all known drugs.
#   No user history needed — just the protein sequence.
# ─────────────────────────────────────────────────────────────────────────────

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from features import morgan_fp, mol_descriptors, amino_acid_composition, AMINO_ACIDS
from config   import THRESHOLD_DAVIS as THRESHOLD, OUTPUT_DIR, FP_BITS


def build_pair_features(smiles: str, sequence: str,
                        scaler, fp_end: int, desc_end: int) -> np.ndarray:
    """Build and scale the feature vector for one drug-protein pair."""
    fp  = morgan_fp(smiles)
    d   = mol_descriptors(smiles)
    aac = amino_acid_composition(sequence)
    x   = np.hstack([fp, d, aac]).reshape(1, -1).astype(np.float32)
    x[:, fp_end:desc_end+20] = scaler.transform(x[:, fp_end:desc_end+20])
    return x


def recommend_drugs(target_sequence: str,
                    drug_library_df: pd.DataFrame,
                    model,
                    scaler,
                    fp_end: int,
                    desc_end: int,
                    top_k: int = 10) -> pd.DataFrame:
    """
    Score every drug in the library against the target protein.
    Return Top-K candidates ranked by predicted pKd.

    Parameters:
        target_sequence  — amino acid sequence of the target protein
        drug_library_df  — DataFrame with column 'smiles'
        model            — trained model with .predict(X) method
        scaler           — fitted StandardScaler from split_scale_balance()
        top_k            — number of recommendations to return

    Returns:
        DataFrame: rank | smiles | predicted_pkd | binding_class | kd_nM
    """
    scores = []
    for smi in drug_library_df["smiles"].unique():
        if Chem.MolFromSmiles(smi) is None:
            continue
        x   = build_pair_features(smi, target_sequence, scaler, fp_end, desc_end)
        pkd = float(model.predict(x)[0])
        kd  = 10 ** (-pkd)
        scores.append({
            "smiles"        : smi,
            "predicted_pkd" : round(pkd, 3),
            "kd_nM"         : round(kd * 1e9, 2),
            "binding_class" : "Binder" if pkd >= THRESHOLD else "Non-binder",
        })

    recs = (pd.DataFrame(scores)
              .sort_values("predicted_pkd", ascending=False)
              .head(top_k)
              .reset_index(drop=True))
    recs.index = recs.index + 1
    recs.index.name = "Rank"
    return recs


def plot_recommendations(recs: pd.DataFrame, title: str = "Top Recommended Drugs"):
    """Bar chart of top-K drug predictions."""
    fig, ax = plt.subplots(figsize=(9, 4))
    labels  = [f"Drug #{i}" for i in recs.index]
    colors  = ["#2E75B6" if b == "Binder" else "#AED6F1"
               for b in recs["binding_class"]]
    ax.barh(labels[::-1], recs["predicted_pkd"][::-1], color=colors[::-1])
    ax.axvline(THRESHOLD, color="red", linestyle="--",
               label=f"Binder threshold (pKd={THRESHOLD})")
    ax.set_xlabel("Predicted pKd")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/recommendations.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"[recommend] Chart saved → {out}")

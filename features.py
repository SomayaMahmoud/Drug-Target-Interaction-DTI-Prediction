# features.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTORS:
#   Person B — TF-IDF character n-grams (creative NLP approach)
#              Original: TfidfVectorizer(analyzer='char', ngram_range=(1,3))
#   Person D — Morgan fingerprints (512 bits) + molecular descriptors (5)
#              + amino acid composition (20), combined into feature matrix
#              + StandardScaler, SMOTE, SelectKBest
#   Person E — Morgan fingerprints (2048 bits), amino acid composition (20),
#              proper scaling (train-fit / test-transform only), precomputing
#              unique-molecule fingerprints for efficiency
#
# What each approach does:
#   Person B  → treat SMILES/protein as TEXT (no chemistry library needed)
#   Person D  → convert chemistry to hand-crafted numbers
#   Person E  → larger fingerprint + cleaner code + vectorised computation
# ─────────────────────────────────────────────────────────────────────────────

import numpy  as np
import pandas as pd
import time, warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling   import SMOTE

from config import (SEED, TEST_SIZE, FP_BITS, FP_RADIUS,
                    TFIDF_FEATURES, TFIDF_NGRAM)

# ── amino acids (Person D + E) ────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# columns where continuous features start/end (after fingerprints)
SCALE_START = FP_BITS          # index 2048
SCALE_END   = FP_BITS + 17     # 12 descriptors + 5 original + ignore overlap = 17
# Simpler: we'll mark them at build time


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH A — Morgan Fingerprints  (Person D + E)
# Person D used 512 bits; Person E used 2048 bits.
# We use 2048 (more information) but keep Person D's 512 available.
# ─────────────────────────────────────────────────────────────────────────────

_fp_gen_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_fp_gen_512  = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=512)

def morgan_fp(smi: str, bits: int = FP_BITS) -> np.ndarray:
    """
    Convert a SMILES string to a Morgan fingerprint bit-vector.

    Person E's version (2048 bits, vectorised):
        Each bit = whether a specific molecular substructure is present (1) or absent (0).
        Radius 2 = ECFP4 (the industry standard).
    Person D's version used 512 bits (faster but less information).
    """
    gen = _fp_gen_2048 if bits == FP_BITS else _fp_gen_512
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(bits, dtype=np.uint8)
    return gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH B — Molecular Descriptors  (Person D — Steps 14-15)
# Person D originally used 5 descriptors, then upgraded to combined features.
# We use 12 validated descriptors (FractionCSP3 works in all RDKit versions).
# ─────────────────────────────────────────────────────────────────────────────

def mol_descriptors(smi: str) -> np.ndarray:
    """
    12 physicochemical properties per drug molecule.
    Person D's original 5: MolWt, LogP, HDonors, HAcceptors, TPSA.
    Extended to 12 here for richer representation.
    All descriptors verified to exist in all RDKit versions.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(12, dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol),              # molecular weight
        Descriptors.MolLogP(mol),            # lipophilicity (fat vs water)
        Descriptors.TPSA(mol),               # polar surface area
        Descriptors.NumHDonors(mol),         # hydrogen bond donors
        Descriptors.NumHAcceptors(mol),      # hydrogen bond acceptors
        Descriptors.NumRotatableBonds(mol),  # molecular flexibility
        Descriptors.HeavyAtomCount(mol),     # non-H atom count
        Descriptors.RingCount(mol),          # total ring count
        Descriptors.NumAromaticRings(mol),   # aromatic rings
        Descriptors.NumHeteroatoms(mol),     # non-C non-H atoms
        Descriptors.FractionCSP3(mol),       # fraction of saturated C
        Lipinski.NumSaturatedRings(mol),     # saturated ring count
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH C — Amino Acid Composition  (Person D — Step 18, Person E)
# Person D's first version counted only 4 amino acids.
# Person D upgraded to full 20-AA composition in Step 18.
# Person E had the same improvement with clean code.
# ─────────────────────────────────────────────────────────────────────────────

def amino_acid_composition(seq: str) -> np.ndarray:
    """
    Person D (Step 18) and Person E's shared approach:
        For each of the 20 standard amino acids, compute its frequency.
        e.g. if protein is 100 aa and has 10 Alanines → A = 0.10
    Person D's earlier version (Step 5) only counted A, C, D + length.
    """
    seq = seq.upper()
    n   = max(len(seq), 1)
    return np.array([seq.count(aa) / n for aa in AMINO_ACIDS], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH D — TF-IDF Character N-Grams  (Person B — ai_project.ipynb)
# Treat SMILES and protein sequences as plain text.
# Character trigrams capture local patterns without any chemistry library.
# This is a clever baseline that requires no domain knowledge.
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_features(df: pd.DataFrame):
    """
    Person B's complete pipeline:
        TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=1000)
        applied to both SMILES and protein sequences.
    Returns X_tfidf of shape (N, 2000).
    """
    tfidf_smiles = TfidfVectorizer(
        analyzer   = "char",
        ngram_range= TFIDF_NGRAM,
        max_features= TFIDF_FEATURES
    )
    tfidf_prot = TfidfVectorizer(
        analyzer   = "char",
        ngram_range= TFIDF_NGRAM,
        max_features= TFIDF_FEATURES
    )

    X_drug = tfidf_smiles.fit_transform(df["smiles"]).toarray()
    X_prot = tfidf_prot.fit_transform(df["sequence"]).toarray()
    X_tfidf = np.hstack([X_drug, X_prot])

    print(f"[features] TF-IDF matrix (Person B): {X_tfidf.shape}")
    return X_tfidf, tfidf_smiles, tfidf_prot


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: build full feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    """
    Combine all features into one matrix.
    Person E's efficient approach: precompute unique molecules first.
    Person D's approach: loop through all rows (slower but identical result).

    Returns:
        X        — (N, 2080) = 2048 FP + 12 descriptors + 20 AAC
        y_reg    — continuous affinity (for regression)
        y_cls    — binary label (for classification)
        fp_end   — column index where fingerprints end (= 2048)
        desc_end — column index where descriptors end (= 2060)
    """
    print("[features] Building Morgan fingerprints (Person D + E approach)…")
    t0 = time.time()

    # Person E's optimisation: precompute unique molecules
    uniq_smiles = df["smiles"].unique()
    fp_cache    = {s: morgan_fp(s)        for s in uniq_smiles}
    desc_cache  = {s: mol_descriptors(s)  for s in uniq_smiles}

    FP   = np.stack([fp_cache[s]   for s in df["smiles"]])
    DESC = np.stack([desc_cache[s] for s in df["smiles"]])
    DESC = np.nan_to_num(DESC, nan=0.0, posinf=0.0, neginf=0.0)

    print("[features] Building amino acid composition (Person D Step 18 + E)…")
    uniq_seqs = df["sequence"].unique()
    aac_cache = {s: amino_acid_composition(s) for s in uniq_seqs}
    AAC = np.stack([aac_cache[s] for s in df["sequence"]])

    X     = np.hstack([FP, DESC, AAC]).astype(np.float32)  # (N, 2080)
    y_reg = df["affinity"].values.astype(np.float32)
    y_cls = df["label"].values.astype(np.int32)

    fp_end   = FP_BITS           # 2048
    desc_end = FP_BITS + 12      # 2060

    print(f"[features] Feature matrix: {X.shape}  |  "
          f"Binders: {y_cls.sum():,} ({100*y_cls.mean():.1f}%)  |  "
          f"Done in {time.time()-t0:.0f}s")
    return X, y_reg, y_cls, fp_end, desc_end


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT + SCALE + BALANCE  (Person D Steps 7/11/16/20 + Person E)
# ─────────────────────────────────────────────────────────────────────────────

def split_scale_balance(X, y_reg, y_cls, fp_end: int, desc_end: int):
    """
    Person D's pipeline corrected:
        1. Stratified split (ensures same imbalance in train & test)
        2. Scale ONLY continuous columns (not binary fingerprints!)
           Person D's original: scaler.fit_transform(X) — this wrongly
           scaled binary 0/1 fingerprints, distorting their meaning.
        3. Sample weights: upweight minority class (binders)
        4. SMOTE on classification labels (not on regression targets!)
           SMOTE cannot handle continuous y — only integer class labels.
        5. SelectKBest (Person D Step 20): reduce to top 300 features
    """

    # ── 1. stratified split ───────────────────────────────────────────────
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_cls,
        test_size=TEST_SIZE, random_state=SEED, stratify=y_cls
    )

    # ── 2. scale ONLY descriptor + AAC columns ────────────────────────────
    scaler = StandardScaler()
    X_tr[:, fp_end:desc_end+20] = scaler.fit_transform(X_tr[:, fp_end:desc_end+20])
    X_te[:, fp_end:desc_end+20] = scaler.transform    (X_te[:, fp_end:desc_end+20])

    # ── 3. sample weights ─────────────────────────────────────────────────
    pos_ratio  = yc_tr.sum() / max(len(yc_tr) - yc_tr.sum(), 1)
    sample_wts = np.where(yc_tr == 1, 1.0, pos_ratio).astype(np.float32)

    # ── 4. SMOTE (Person D Steps 11, 19) ─────────────────────────────────
    try:
        sm = SMOTE(random_state=SEED, k_neighbors=3)
        X_bal, yc_bal = sm.fit_resample(X_tr, yc_tr)
        print(f"[features] SMOTE: {len(X_tr):,} → {len(X_bal):,} samples")
    except Exception as e:
        X_bal, yc_bal = X_tr, yc_tr
        print(f"[features] SMOTE skipped ({e}) — using sample_weight instead")

    # ── 5. SelectKBest (Person D Step 20) ────────────────────────────────
    selector = SelectKBest(score_func=f_classif, k=300)
    X_sel_tr = selector.fit_transform(X_bal, yc_bal)
    X_sel_te = selector.transform(X_te)
    print(f"[features] SelectKBest: {X_bal.shape[1]} → {X_sel_tr.shape[1]} features")

    print(f"[features] Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    return dict(
        X_train     = X_tr,
        X_test      = X_te,
        y_train_reg = yr_tr,
        y_test_reg  = yr_te,
        y_train_cls = yc_tr,
        y_test_cls  = yc_te,
        X_balanced  = X_bal,
        y_bal_cls   = yc_bal,
        X_sel_train = X_sel_tr,
        X_sel_test  = X_sel_te,
        sample_wts  = sample_wts,
        scaler      = scaler,
        selector    = selector,
        fp_end      = fp_end,
        desc_end    = desc_end,
    )


# standalone test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import load_dataset
    df = load_dataset("davis_all.csv", "davis")
    X, y_reg, y_cls, fp_end, desc_end = build_features(df)
    splits = split_scale_balance(X, y_reg, y_cls, fp_end, desc_end)
    print("X_train shape:", splits["X_train"].shape)
    X_tfidf, _, _ = build_tfidf_features(df)

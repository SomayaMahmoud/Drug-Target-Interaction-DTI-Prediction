# config.py — shared settings for the whole project
# ─────────────────────────────────────────────────
import os

# paths — update DATA_PATH to wherever your CSV lives
DATA_PATH_DAVIS = "davis_all.csv"
DATA_PATH_KIBA  = "kiba_all.csv"
OUTPUT_DIR      = "outputs"

# experiment
SEED        = 42
TEST_SIZE   = 0.20

# affinity thresholds for binary label
THRESHOLD_DAVIS = 7.0    # pKd ≥ 7  → binder  (used by Person D)
THRESHOLD_KIBA  = 12.1   # KIBA ≥ 12.1 → binder (used by Person D)

# feature sizes
FP_BITS    = 2048   # Morgan fingerprint length  (Person E used 2048, Person D used 512)
FP_BITS_D  = 512    # Person D's original shorter version kept for comparison
FP_RADIUS  = 2      # ECFP4

# TF-IDF (Person B)
TFIDF_FEATURES = 1000
TFIDF_NGRAM    = (1, 3)

# LSTM input lengths (Person A: 100/500 | Person C: 120/300)
SMILES_MAX_LEN   = 150   # covers >99% of SMILES strings
PROTEIN_MAX_LEN  = 800   # covers >95% of sequences
EMBED_DIM        = 32

# deep learning
DL_EPOCHS     = 20
DL_BATCH_SIZE = 64
DL_LR         = 0.001

# make output dir on import
os.makedirs(OUTPUT_DIR, exist_ok=True)

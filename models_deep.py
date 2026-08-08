# models_deep.py
# ─────────────────────────────────────────────────────────────────────────────
# CONTRIBUTORS:
#   Person A — Keras LSTM with Tokenizer+Embedding for SMILES (maxlen=100)
#               and protein sequences (maxlen=500)
#               Dataset: Davis (sampled 5000 for speed)
#
#   Person C — Keras LSTM (improved over Person A):
#               • GPU mixed precision (tf.keras.mixed_precision)
#               • Target normalisation (z-score before training)
#               • Full 30000 samples (not a subset)
#               • Shorter maxlen: SMILES=120, protein=300
#
#   Person E — PyTorch MLP (DTAPredictor class):
#               Input→1024→512→256→128→1
#               BatchNorm + Dropout + Adam + MSE loss
#               Standard 80/20 split, early-stopping-aware training loop
#
# All three are preserved as separate functions.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Input,
                                      Concatenate, Dropout, BatchNormalization)
from tensorflow.keras.models    import Model
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (SEED, DL_EPOCHS, DL_BATCH_SIZE, DL_LR,
                    SMILES_MAX_LEN, PROTEIN_MAX_LEN, EMBED_DIM)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# PERSON A — Keras LSTM (notebook5a470638f5.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_personA(vocab_smiles: int, vocab_prot: int,
                       smiles_len: int = 100, prot_len: int = 500):
    """
    Person A's exact architecture:
        Input(SMILES) → Embedding(vocab,32) → LSTM(32) ─┐
        Input(Prot)   → Embedding(vocab,32) → LSTM(32) ─┤→ Concatenate → Dense(64,relu) → Dense(1)
    """
    input_s = Input(shape=(smiles_len,), name="smiles")
    x1 = Embedding(input_dim=vocab_smiles, output_dim=32)(input_s)
    x1 = LSTM(32)(x1)

    input_p = Input(shape=(prot_len,), name="protein")
    x2 = Embedding(input_dim=vocab_prot, output_dim=32)(input_p)
    x2 = LSTM(32)(x2)

    merged = Concatenate()([x1, x2])
    out    = Dense(64, activation="relu")(merged)
    out    = Dense(1)(out)

    model = Model([input_s, input_p], out)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_personA(df, epochs=10):
    """
    Person A's training pipeline.
    Uses char-level Tokenizer on SMILES (maxlen=100) and protein (maxlen=500).
    Samples 5000 rows for speed (as in Person A's original notebook).
    """
    print("[deep] Training LSTM — Person A architecture…")
    sample = df.sample(min(5000, len(df)), random_state=SEED)

    tok_s = Tokenizer(char_level=True); tok_s.fit_on_texts(sample["smiles"])
    tok_p = Tokenizer(char_level=True); tok_p.fit_on_texts(sample["sequence"])

    S = pad_sequences(tok_s.texts_to_sequences(sample["smiles"]),   maxlen=100)
    P = pad_sequences(tok_p.texts_to_sequences(sample["sequence"]), maxlen=500)
    y = sample["affinity"].values

    from sklearn.model_selection import train_test_split
    Str, Ste, Ptr, Pte, ytr, yte = train_test_split(S, P, y, test_size=0.2, random_state=SEED)

    model = build_lstm_personA(len(tok_s.word_index)+1, len(tok_p.word_index)+1)
    model.fit([Str, Ptr], ytr, epochs=epochs, batch_size=64,
              validation_split=0.1, verbose=1)

    preds = model.predict([Ste, Pte]).flatten()
    return model, preds, yte, tok_s, tok_p


# ─────────────────────────────────────────────────────────────────────────────
# PERSON C — Keras LSTM (improved, notebook86814afd91.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm_personC(df, epochs=10):
    """
    Person C's improvements over Person A:
      • Mixed precision for GPU speed
      • Target z-score normalisation (improves convergence)
      • Full dataset (30000 samples)
      • Shorter protein maxlen (300 vs 500) — trains faster
      • BatchNormalization layer added
    """
    print("[deep] Training LSTM — Person C architecture (improved)…")

    # Person C: GPU mixed precision
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass  # skip if not available

    smiles   = df["smiles"].astype(str)
    proteins = df["sequence"].astype(str)
    y        = df["affinity"].values

    # Person C: z-score normalisation of targets
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    tok_s = Tokenizer(char_level=True); tok_s.fit_on_texts(smiles)
    tok_p = Tokenizer(char_level=True); tok_p.fit_on_texts(proteins)

    MAX_S, MAX_P = 120, 300   # Person C's lengths
    X_s = pad_sequences(tok_s.texts_to_sequences(smiles),   maxlen=MAX_S)
    X_p = pad_sequences(tok_p.texts_to_sequences(proteins), maxlen=MAX_P)

    from sklearn.model_selection import train_test_split
    Str, Ste, Ptr, Pte, ytr, yte_norm, yte = train_test_split(
        X_s, X_p, y_norm, y, test_size=0.2, random_state=SEED
    )

    # Person C's model (extended Person A with BatchNorm)
    vs = len(tok_s.word_index) + 1
    vp = len(tok_p.word_index) + 1

    inp_s = Input(shape=(MAX_S,)); e_s = Embedding(vs, EMBED_DIM)(inp_s); l_s = LSTM(64)(e_s)
    inp_p = Input(shape=(MAX_P,)); e_p = Embedding(vp, EMBED_DIM)(inp_p); l_p = LSTM(64)(e_p)

    merged = Concatenate()([l_s, l_p])
    x = Dense(128, activation="relu")(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, dtype="float32")(x)   # float32 output for mixed precision

    model = Model([inp_s, inp_p], output)
    model.compile(optimizer="adam", loss="mse")
    model.fit([Str, Ptr], ytr, epochs=epochs, batch_size=64,
              validation_split=0.1, verbose=1)

    # denormalise predictions back to pKd scale
    preds_norm = model.predict([Ste, Pte]).flatten()
    preds = preds_norm * y_std + y_mean

    # reset precision policy
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass

    return model, preds, yte, tok_s, tok_p


# ─────────────────────────────────────────────────────────────────────────────
# PERSON E — PyTorch MLP  (davis_dta_full_project Phase 2b)
# ─────────────────────────────────────────────────────────────────────────────

class DTAPredictor(nn.Module):
    """
    Person E's DTAPredictor:
        Input (2080) → 1024 → 512 → 256 → 128 → 1
        Each hidden layer: Linear → BatchNorm → ReLU → Dropout

    Original Person E used input_dim=2068 (2048 FP + 20 AAC).
    We use 2080 (2048 FP + 12 DESC + 20 AAC = 2080).
    The class auto-detects input size.
    """
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Block 1: input → 1024
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Block 4: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_mlp_personE(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, epochs: int = DL_EPOCHS) -> np.ndarray:
    """
    Person E's training loop.
    Uses MSE loss, Adam optimiser, batch training.
    Returns predicted pKd values for the test set.
    """
    print(f"[deep] Training PyTorch MLP — Person E  (device: {DEVICE})…")

    Xtr = torch.tensor(X_train.astype(np.float32))
    ytr = torch.tensor(y_train.astype(np.float32))
    Xte = torch.tensor(X_test.astype(np.float32)).to(DEVICE)

    loader  = DataLoader(TensorDataset(Xtr, ytr), batch_size=DL_BATCH_SIZE, shuffle=True)
    model   = DTAPredictor(input_dim=X_train.shape[1]).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=DL_LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(Xte).cpu().numpy()

    return model, preds


# ─────────────────────────────────────────────────────────────────────────────
# Train all deep models and return predictions
# ─────────────────────────────────────────────────────────────────────────────

def train_all_deep(df, splits: dict) -> tuple:
    """
    Train all 3 deep learning models.
    Returns (predictions_dict, models_dict) where predictions are on the test set.
    """
    X_tr  = splits["X_train"]
    X_te  = splits["X_test"]
    yr_tr = splits["y_train_reg"]
    yr_te = splits["y_test_reg"]

    preds  = {}
    models = {}

    # ── Person A ─────────────────────────────────────────────────────────────
    try:
        m_a, yp_a, yte_a, _, _ = train_lstm_personA(df)
        # Note: Person A uses its own internal split on a 5000-sample subset
        # so yte_a is the actual y_test for that model
        preds["Keras LSTM (Person A)"]  = (yp_a, yte_a)
        models["Keras LSTM (Person A)"] = m_a
    except Exception as e:
        print(f"  [WARN] Person A LSTM failed: {e}")

    # ── Person C ─────────────────────────────────────────────────────────────
    try:
        m_c, yp_c, yte_c, _, _ = train_lstm_personC(df)
        preds["Keras LSTM (Person C)"]  = (yp_c, yte_c)
        models["Keras LSTM (Person C)"] = m_c
    except Exception as e:
        print(f"  [WARN] Person C LSTM failed: {e}")

    # ── Person E ─────────────────────────────────────────────────────────────
    m_e, yp_e = train_mlp_personE(X_tr, yr_tr, X_te)
    preds["PyTorch MLP (Person E)"]  = (yp_e, yr_te)
    models["PyTorch MLP (Person E)"] = m_e

    return preds, models


# standalone test
if __name__ == "__main__":
    print("models_deep.py: import and call train_all_deep(df, splits)")

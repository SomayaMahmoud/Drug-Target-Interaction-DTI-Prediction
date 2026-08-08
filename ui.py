# ui.py — Gradio interactive demo
# ─────────────────────────────────────────────────────────────────────────────
# Two tabs:
#   Tab 1: Predict binding between ONE drug + ONE protein
#   Tab 2: Recommend TOP-K drugs for a given protein
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
import numpy  as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from features import morgan_fp, mol_descriptors, amino_acid_composition
from recommend import recommend_drugs, build_pair_features
from config   import THRESHOLD_DAVIS as THRESHOLD, FP_BITS

EXAMPLE_DRUGS = {
    "Imatinib (cancer)": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cnnc3ccccc23)n1",
    "Erlotinib (EGFR)" : "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "Aspirin"          : "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine"         : "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
}

EXAMPLE_PROTEIN = (
    "MVSYWDTGVLLCALLSCLLLTGSSSGSKLKDPELSLKGTQHIMQAGQTLHLQ"
    "CRGEAAHKWSLPEMVSKESERLSITKSACGRNGKQFCSTLTLNTAQANHTGFY"
)


def launch_ui(best_model, scaler, drug_library_df,
              fp_end: int, desc_end: int):
    """Build and launch the Gradio web interface."""

    def predict_pair(smiles, sequence):
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return "❌ Invalid SMILES string.", "", ""
        if len(sequence.strip()) < 10:
            return "❌ Protein sequence too short (min 10 aa).", "", ""

        x   = build_pair_features(smiles.strip(),
                                   sequence.strip().upper(),
                                   scaler, fp_end, desc_end)
        pkd = round(float(best_model.predict(x)[0]), 3)
        kd  = 10 ** (-pkd)
        kd_s = f"{kd*1e9:.2f} nM" if kd < 1e-6 else f"{kd*1e6:.2f} µM"
        lbl  = "🟢 BINDER" if pkd >= THRESHOLD else "🔴 NON-BINDER"

        info = (
            f"Predicted pKd   : {pkd}\n"
            f"Kd              : {kd_s}\n"
            f"Classification  : {lbl}  (threshold = {THRESHOLD})\n\n"
            f"Interpretation:\n"
            f"  pKd ≥ 9  → very strong (< 1 nM)\n"
            f"  pKd 7–9  → strong / drug-like (1–100 nM)\n"
            f"  pKd 5–7  → weak (0.1–10 µM)\n"
            f"  pKd < 5  → non-binder (> 10 µM)"
        )
        return info, str(pkd), lbl

    def recommend(sequence, top_k):
        if len(sequence.strip()) < 10:
            return None
        recs = recommend_drugs(
            target_sequence = sequence.strip().upper(),
            drug_library_df = drug_library_df,
            model           = best_model,
            scaler          = scaler,
            fp_end          = fp_end,
            desc_end        = desc_end,
            top_k           = int(top_k),
        )
        return recs.reset_index()[["Rank","smiles","predicted_pkd","kd_nM","binding_class"]]

    with gr.Blocks(title="DTI Predictor — Team Project") as demo:
        gr.Markdown(
            "## 💊 Drug–Target Interaction Predictor\n"
            "Combined team project — Davis Dataset · XGBoost + LSTM + MLP"
        )

        with gr.Tab("🔬 Predict Binding"):
            with gr.Row():
                smi_box = gr.Textbox(label="Drug SMILES", lines=2,
                                     placeholder="Paste SMILES string…")
                seq_box = gr.Textbox(label="Protein Sequence", lines=4,
                                     placeholder="Paste amino acid sequence…")

            drug_dd = gr.Dropdown(label="Example drug",
                                  choices=list(EXAMPLE_DRUGS.keys()))
            drug_dd.change(lambda k: EXAMPLE_DRUGS[k], drug_dd, smi_box)
            gr.Button("Load example protein").click(
                lambda: EXAMPLE_PROTEIN, None, seq_box)

            pred_btn = gr.Button("🔬 Predict", variant="primary")

            with gr.Row():
                out_info  = gr.Textbox(label="Full Details", lines=10)
                out_pkd   = gr.Textbox(label="pKd")
                out_label = gr.Textbox(label="Classification")

            pred_btn.click(predict_pair, [smi_box, seq_box],
                           [out_info, out_pkd, out_label])

        with gr.Tab("📋 Recommend Drugs"):
            gr.Markdown(
                "Enter a target protein sequence.\n"
                "The model will score all known drugs and return the Top-K candidates."
            )
            rec_seq = gr.Textbox(label="Target Protein Sequence",
                                  lines=4, value=EXAMPLE_PROTEIN)
            rec_k   = gr.Slider(label="Top K", minimum=5, maximum=30,
                                 value=10, step=1)
            rec_btn = gr.Button("📋 Recommend", variant="primary")
            rec_out = gr.Dataframe(label="Top Drug Candidates")
            rec_btn.click(recommend, [rec_seq, rec_k], rec_out)

    print("[ui] Launching Gradio — click the URL below…")
    demo.launch(share=False, inbrowser=True)
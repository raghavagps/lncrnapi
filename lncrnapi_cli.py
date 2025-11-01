#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LncRNA–Protein Interaction Prediction CLI
- rapid: composition-based features
- dnabert2: LLM-based embeddings (DNABERT2 + ESM2)
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
from itertools import product
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


# --- FASTA Parser ---
def read_fasta(filepath):
    with open(filepath) as f:
        seqs, seq, header = [], [], None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and seq:
                    seqs.append((header, "".join(seq).upper()))
                    seq = []
                header = line[1:].strip()
            else:
                seq.append(line)
        if header and seq:
            seqs.append((header, "".join(seq).upper()))
    return seqs


# --- Rapid (composition-based) ---
def rapid_percentage_vectorized(sequences, alphabet, prefix):
    seq_series = pd.Series(sequences)
    seq_len = seq_series.str.len().replace(0, 1)
    df = pd.concat(
        [(seq_series.str.count(base) / seq_len * 100).rename(f"{prefix}_{base}") for base in alphabet],
        axis=1
    )
    return df


# --- LLM embedding utilities ---
def get_device():
    if torch.cuda.is_available():
        print("✅ Using GPU (CUDA)")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✅ Using Apple MPS GPU")
        return torch.device("mps")
    else:
        print("⚠️ Using CPU (may be slow)")
        return torch.device("cpu")


def generate_embedding(sequence, tokenizer, model, device):
    tokens = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.cpu().numpy()[0, 1:-1, :].mean(axis=0)
    return embedding


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Predict lncRNA–protein interaction probability using CatBoost model (rapid or dnabert2, all-by-all)."
    )
    parser.add_argument("-lf", required=True, help="Path to lncRNA FASTA file")
    parser.add_argument("-pf", required=True, help="Path to protein FASTA file")
    parser.add_argument("-wd", required=True, help="Working/output directory")
    parser.add_argument("-model", required=True, choices=["rapid", "dnabert2"],
                        help="Select which model to use: 'rapid' or 'dnabert2'")
    parser.add_argument("-t", type=float, default=0.5, help="Threshold for classification (default=0.5)")
    args = parser.parse_args()

    wd = Path(args.wd)
    wd.mkdir(parents=True, exist_ok=True)
    output_path = wd / "output.csv"

    # --- Load FASTA files ---
    lncrnas = read_fasta(args.lf)
    proteins = read_fasta(args.pf)
    print(f"Loaded {len(lncrnas)} lncRNAs and {len(proteins)} proteins.\n")

    lnc_ids = [i for i, _ in lncrnas]
    prot_ids = [i for i, _ in proteins]
    lnc_seqs = [s.replace("U", "T") for _, s in lncrnas]
    prot_seqs = [s for _, s in proteins]

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(
        model_dir,
        "catboost_model_rapid.joblib" if args.model == "rapid" else "catboost_dnabert2_esm-t30.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Using model: {args.model.upper()} → {model_path}\n")

    # --- Feature Generation ---
    if args.model == "rapid":
        print("[Step 1] Generating composition-based features...")
        lnc_df = rapid_percentage_vectorized(lnc_seqs, list("ATGC"), prefix="CDK")
        prot_df = rapid_percentage_vectorized(prot_seqs, list("GQYNRLCWTV"), prefix="AAC")

        pairs = list(product(range(len(lncrnas)), range(len(proteins))))
        print(f"Generating {len(pairs)} lncRNA–protein pairs...")

        lnc_idx = [i for i, _ in pairs]
        prot_idx = [j for _, j in pairs]
        X = pd.concat(
            [lnc_df.iloc[lnc_idx].reset_index(drop=True),
             prot_df.iloc[prot_idx].reset_index(drop=True)],
            axis=1
        )

    elif args.model == "dnabert2":
        print("[Step 1] Loading DNABERT2 and ESM2 models from Hugging Face...")
        device = get_device()
        lncrna_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
        lncrna_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M").to(device).eval()

        protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        protein_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device).eval()

        print("[Step 2] Generating embeddings (this may take a while)...")
        lnc_embs = {
            l_id: generate_embedding(seq, lncrna_tokenizer, lncrna_model, device)
            for l_id, seq in tqdm(zip(lnc_ids, lnc_seqs), total=len(lnc_ids), desc="LncRNA embeddings")
        }
        prot_embs = {
            p_id: generate_embedding(seq, protein_tokenizer, protein_model, device)
            for p_id, seq in tqdm(zip(prot_ids, prot_seqs), total=len(prot_ids), desc="Protein embeddings")
        }

        pairs = list(product(lnc_ids, prot_ids))
        X = [
            np.concatenate([lnc_embs[l_id], prot_embs[p_id]])
            for l_id, p_id in tqdm(pairs, desc="Building feature vectors")
        ]
        X = np.stack(X)

    # --- Load Model ---
    print("\n[Step 3] Loading CatBoost model...")
    model = joblib.load(model_path)

    # --- Predict ---
    print("[Step 4] Predicting interaction probabilities...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= args.t).astype(int)

    # --- Save Output ---
    results = pd.DataFrame({
        "lncRNA_ID": [lnc_ids[i] if isinstance(i, int) else i for i, _ in pairs],
        "Protein_ID": [prot_ids[j] if isinstance(j, int) else j for _, j in pairs],
        "Interaction_Probability": probs,
        "Predicted_Label": preds
    })
    results.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to {output_path}")
    print(results.head())


if __name__ == "__main__":
    main()

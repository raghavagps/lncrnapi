#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LncRNA-Protein Interaction Prediction CLI

This script takes an lncRNA FASTA file and a protein FASTA file, generates
embeddings using pre-trained language models (DNABERT-2 and ESM-2), and
predicts the interaction probability using a pre-trained CatBoost model.

Usage:
    python predict_interaction.py \
        --lncrna_fasta /path/to/your/lncrnas.fasta \
        --protein_fasta /path/to/your/proteins.fasta \
        --model_path /path/to/your/saved_model.joblib \
        --output_file /path/to/results.csv
"""
import os
import argparse
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier

# Suppress informational warnings from the transformers package
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModel

# --- MODEL CONFIGURATION ---
LNCRNA_MODEL_NAME = "zhihan1996/DNABERT-2-117M"
PROTEIN_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"


def parse_fasta(file_path):
    """Parses a FASTA file and returns a dictionary of sequences."""
    sequences = {}
    current_id = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:].split()[0] # Get ID, remove extra info
                sequences[current_id] = ''
            elif current_id:
                sequences[current_id] += line
    if not sequences:
        raise ValueError(f"No sequences found in FASTA file: {file_path}")
    return sequences

def get_device():
    """Checks for available hardware and returns the appropriate torch device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using GPU (CUDA) for acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU not found. Using CPU. This might be slow.")
    return device

def generate_embedding(sequence, tokenizer, model, device):
    """
    Generates a single embedding for a given sequence using the provided
    transformer model and tokenizer.
    """
    # Tokenize the sequence
    tokens = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Get model output (hidden states)
    with torch.no_grad():
        output = model(**tokens)

    # Calculate the mean of the last hidden state to get a fixed-size embedding
    # We ignore the special tokens [CLS] and [SEP] by selecting [1:-1]
    embedding = output.last_hidden_state.cpu().numpy()[0, 1:-1, :].mean(axis=0)
    return embedding

def load_catboost_model(model_path):
    """Loads a CatBoost model from a file, supporting multiple formats."""
    print(f"\n[Step 4] Loading CatBoost model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if model_path.endswith('.cbm'):
        model = CatBoostClassifier()
        model.load_model(model_path)
    elif model_path.endswith('.joblib'):
        # This is the logic that handles your joblib-saved model
        print("  - Detected .joblib file, using joblib.load().")
        model = joblib.load(model_path)
    elif model_path.endswith('.pkl'):
        print("  - Detected .pkl file, using pickle.load().")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError("Unsupported model format. Use .cbm, .joblib, or .pkl")

    print("✅ CatBoost model loaded successfully.")
    return model

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(
        description="Predict lncRNA-Protein Interaction Probability.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--lncrna_fasta",
        type=str,
        required=True,
        help="Path to the FASTA file containing lncRNA sequences."
    )
    parser.add_argument(
        "--protein_fasta",
        type=str,
        required=True,
        help="Path to the FASTA file containing protein sequences."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved CatBoost model file (.cbm, .joblib, or .pkl)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file where results will be saved."
    )
    args = parser.parse_args()

    # --- Step 0: Setup ---
    device = get_device()

    # --- Step 1: Load Language Models and Tokenizers ---
    print("\n[Step 1] Loading language models from Hugging Face...")
    try:
        # lncRNA Model (DNABERT-2)
        lncrna_tokenizer = AutoTokenizer.from_pretrained(LNCRNA_MODEL_NAME)
        # Add trust_remote_code=True to automatically accept custom model code
        lncrna_model = AutoModel.from_pretrained(LNCRNA_MODEL_NAME).to(device)
        lncrna_model.eval() # Set model to evaluation mode
        print(f"  - Loaded {LNCRNA_MODEL_NAME}")

        # Protein Model (ESM-2)
        protein_tokenizer = AutoTokenizer.from_pretrained(PROTEIN_MODEL_NAME)
        protein_model = AutoModel.from_pretrained(PROTEIN_MODEL_NAME).to(device)
        protein_model.eval() # Set model to evaluation mode
        print(f"  - Loaded {PROTEIN_MODEL_NAME}")
    except Exception as e:
        error_msg = str(e)
        # Catch the specific security-related error and provide a helpful message
        if "safetensors" in error_msg or "torch.load" in error_msg:
            print(f"❌ Error loading models due to a security restriction: {e}")
            print("\n💡 This is often solved by installing the 'safetensors' library, which allows for secure model loading.")
            print("   Please run the following command in your activated conda/virtual environment:")
            print("   pip install safetensors")
        else:
            print(f"❌ Error loading models: {e}")
            print("   Please ensure you have an active internet connection and the model names are correct.")
        return

    # --- Step 2: Parse Input FASTA Files ---
    print("\n[Step 2] Parsing FASTA files...")
    try:
        lncrna_seqs = parse_fasta(args.lncrna_fasta)
        protein_seqs = parse_fasta(args.protein_fasta)
        print(f"  - Found {len(lncrna_seqs)} lncRNA sequence(s).")
        print(f"  - Found {len(protein_seqs)} protein sequence(s).")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error parsing FASTA files: {e}")
        return

    # --- Step 3: Generate Embeddings ---
    print("\n[Step 3] Generating embeddings for all sequences...")
    lncrna_embeddings = {
        seq_id: generate_embedding(seq, lncrna_tokenizer, lncrna_model, device)
        for seq_id, seq in tqdm(lncrna_seqs.items(), desc="LncRNA Embeddings")
    }
    protein_embeddings = {
        seq_id: generate_embedding(seq, protein_tokenizer, protein_model, device)
        for seq_id, seq in tqdm(protein_seqs.items(), desc="Protein Embeddings")
    }
    print("✅ Embeddings generated.")

    # --- Step 4: Load Classifier and Predict ---
    try:
        classifier = load_catboost_model(args.model_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
        return

    print("\n[Step 5] Predicting interaction probabilities for all pairs...")
    results = []
    total_pairs = len(lncrna_seqs) * len(protein_seqs)
    print(f"  - Creating and evaluating {total_pairs} pair(s).")

    all_pairs = [
        (lnc_id, lnc_emb, prot_id, prot_emb)
        for lnc_id, lnc_emb in lncrna_embeddings.items()
        for prot_id, prot_emb in protein_embeddings.items()
    ]

    for lnc_id, lnc_emb, prot_id, prot_emb in tqdm(all_pairs, desc="Predicting Pairs"):
        # Concatenate embeddings to create the feature vector
        feature_vector = np.concatenate([lnc_emb, prot_emb]).reshape(1, -1)

        # Predict probability [prob_class_0, prob_class_1]
        probability = classifier.predict_proba(feature_vector)[0][1]

        results.append({
            "LncRNA_ID": lnc_id,
            "Protein_ID": prot_id,
            "Interaction_Probability": probability
        })

    # --- Step 6: Save and Display Results ---
    results_df = pd.DataFrame(results)
    # Format probability for better readability
    results_df['Interaction_Probability'] = results_df['Interaction_Probability'].map('{:.4f}'.format)
    
    print(f"\n[Step 6] Saving results to {args.output_file}...")
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir: # Check if the path includes a directory
            os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(args.output_file, index=False)
        print(f"✅ Results successfully saved.")

    except Exception as e:
        print(f"❌ Error saving results to file: {e}")

    # Also print the results to the console for immediate feedback
    print("\n--- Prediction Results ---")
    print(results_df.to_string(index=False))
    print("--------------------------\n")

if __name__ == "__main__":
    main()



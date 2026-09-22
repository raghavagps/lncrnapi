#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified LncRNA–Protein Interaction Prediction CLI

Supported prediction methods
----------------------------
1. rapid
   Composition-based model using:
       - lncRNA CDK: 4 nucleotide composition features
       - Protein AAC: 10 amino-acid composition features

2. dacc_esm2_t30
   Existing model using:
       - lncRNA DACC: 288 features
       - Protein ESM-2 t30: 640 features
       - Total: 928 features

Both methods perform all-by-all lncRNA–protein prediction.

The interaction label is assigned using the user-defined probability
threshold:
    probability >= threshold -> Interacting
    probability < threshold  -> Non-interacting
"""

import os
import sys
import argparse
import subprocess
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------
# Existing DACC + ESM-2 t30 model
# ------------------------------------------------------------

DACC_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "calc_DACC.py"
)

DACC_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "model",
    "catboost_final_model_full.cbm"
)


# ------------------------------------------------------------
# Rapid model
# ------------------------------------------------------------

RAPID_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "model",
    "catboost_model_rapid.joblib"
)


# ------------------------------------------------------------
# Protein language model
# ------------------------------------------------------------

PROTEIN_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"


# ------------------------------------------------------------
# Expected feature dimensions
# ------------------------------------------------------------

EXPECTED_DACC_FEATURES = 288
EXPECTED_ESM_FEATURES = 640
EXPECTED_DACC_ESM_FEATURES = 928

EXPECTED_RAPID_LNCRNA_FEATURES = 4
EXPECTED_RAPID_PROTEIN_FEATURES = 10
EXPECTED_RAPID_FEATURES = 14


# ------------------------------------------------------------
# Rapid model alphabets
# ------------------------------------------------------------

RAPID_LNCRNA_ALPHABET = list("ATGC")

RAPID_PROTEIN_ALPHABET = list(
    "GQYNRLCWTV"
)


# ------------------------------------------------------------
# DACC physicochemical properties
# ------------------------------------------------------------

DACC_PROPERTIES = [
    "p1", "p2", "p3", "p4",
    "p5", "p6", "p7", "p8",
    "p9", "p10", "p11", "p12"
]


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# FASTA parser
# ============================================================

def read_fasta(fasta_file):
    """
    Read sequences from a FASTA file.

    Returns
    -------
    dict
        Dictionary mapping sequence IDs to sequences.
    """

    sequences = {}

    sequence_id = None
    sequence = []

    with open(fasta_file, "r") as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if sequence_id is not None:
                    sequences[sequence_id] = "".join(sequence)

                sequence_id = line[1:].split()[0]
                sequence = []

            else:

                sequence.append(line)

        if sequence_id is not None:
            sequences[sequence_id] = "".join(sequence)

    return sequences


# ============================================================
# Device selection
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Rapid model feature generation
# ============================================================

def generate_composition_features(
    sequences,
    alphabet,
    prefix
):
    """
    Generate percentage composition features.

    Parameters
    ----------
    sequences : list
        Input sequences.

    alphabet : list
        Characters to calculate composition for.

    prefix : str
        Prefix for feature names.

    Returns
    -------
    pandas.DataFrame
        Composition feature matrix.
    """

    seq_series = pd.Series(sequences)

    seq_len = (
        seq_series
        .str.len()
        .replace(0, 1)
    )

    feature_df = pd.concat(
        [
            (
                seq_series.str.count(base)
                / seq_len
                * 100
            ).rename(
                f"{prefix}_{base}"
            )
            for base in alphabet
        ],
        axis=1
    )

    return feature_df


def generate_rapid_features(
    lncrna_sequences,
    protein_sequences
):
    """
    Generate all-by-all rapid model feature matrix.

    lncRNA:
        CDK composition = A, T, G, C

    Protein:
        AAC composition = G, Q, Y, N, R, L, C, W, T, V

    Total:
        4 + 10 = 14 features.
    """

    logger.info(
        "Generating rapid composition-based features..."
    )

    # Rapid model uses DNA alphabet.
    # Convert RNA U to T for lncRNA composition.
    lncrna_sequences = [
        sequence.upper().replace("U", "T")
        for sequence in lncrna_sequences
    ]

    protein_sequences = [
        sequence.upper()
        for sequence in protein_sequences
    ]

    lnc_features = generate_composition_features(
        lncrna_sequences,
        RAPID_LNCRNA_ALPHABET,
        prefix="CDK"
    )

    protein_features = generate_composition_features(
        protein_sequences,
        RAPID_PROTEIN_ALPHABET,
        prefix="AAC"
    )

    # Build all-by-all feature matrix without creating
    # an unnecessary list of pair tuples.
    lnc_count = len(lncrna_sequences)
    protein_count = len(protein_sequences)

    lnc_indices = np.repeat(
        np.arange(lnc_count),
        protein_count
    )

    protein_indices = np.tile(
        np.arange(protein_count),
        lnc_count
    )

    X = pd.concat(
        [
            lnc_features.iloc[
                lnc_indices
            ].reset_index(drop=True),

            protein_features.iloc[
                protein_indices
            ].reset_index(drop=True)
        ],
        axis=1
    )

    if X.shape[1] != EXPECTED_RAPID_FEATURES:

        raise ValueError(
            f"Rapid model generated {X.shape[1]} "
            f"features, but {EXPECTED_RAPID_FEATURES} "
            f"features are expected."
        )

    return X


# ============================================================
# ESM-2 model loading
# ============================================================

def load_esm_model(device):

    logger.info(
        f"Loading protein language model: "
        f"{PROTEIN_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        PROTEIN_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        PROTEIN_MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


# ============================================================
# Protein embedding
# ============================================================

def generate_protein_embedding(
    sequence,
    tokenizer,
    model,
    device
):
    """
    Generate a 640-dimensional ESM-2 t30 embedding.

    Pooling:
        output.last_hidden_state[0, 1:-1, :].mean(axis=0)
    """

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model(**inputs)

    embedding = (
        output.last_hidden_state
        .cpu()
        .numpy()[0, 1:-1, :]
        .mean(axis=0)
    )

    if embedding.shape[0] != EXPECTED_ESM_FEATURES:

        raise ValueError(
            f"Unexpected ESM-2 embedding dimension: "
            f"{embedding.shape[0]}. "
            f"Expected {EXPECTED_ESM_FEATURES}."
        )

    return embedding


# ============================================================
# DACC calculation
# ============================================================

def calculate_dacc(
    lncrna_fasta,
    dacc_output
):
    """
    Run calc_DACC.py using the user-provided lncRNA FASTA.
    """

    if not os.path.isfile(DACC_SCRIPT):

        raise FileNotFoundError(
            f"calc_DACC.py was not found at:\n"
            f"{DACC_SCRIPT}"
        )

    logger.info(
        "Calculating DACC features..."
    )

    command = [
        sys.executable,
        DACC_SCRIPT,

        "-i",
        lncrna_fasta,

        "-o",
        dacc_output,

        "-p",
        *DACC_PROPERTIES
    ]

    logger.info(
        "Running: " + " ".join(command)
    )

    subprocess.run(
        command,
        check=True
    )

    if not os.path.isfile(dacc_output):

        raise FileNotFoundError(
            "DACC calculation completed, but the "
            f"expected output file was not found:\n"
            f"{dacc_output}"
        )

    logger.info(
        f"DACC features written to: "
        f"{dacc_output}"
    )


# ============================================================
# Load DACC features
# ============================================================

def load_dacc_features(dacc_file):
    """
    Load DACC features from the generated CSV.

    Expected:
        288 DACC features per lncRNA.
    """

    df = pd.read_csv(dacc_file)

    if df.empty:

        raise ValueError(
            "DACC output file is empty."
        )

    # Identify sequence ID column
    possible_id_columns = [
        "Sequence_ID",
        "sequence_id",
        "SequenceID",
        "ID",
        "id"
    ]

    id_column = None

    for column in possible_id_columns:

        if column in df.columns:

            id_column = column
            break

    if id_column is None:

        raise ValueError(
            "Could not identify the sequence ID column "
            "in the DACC output."
        )

    # Everything except the ID column is treated
    # as a DACC feature.
    feature_columns = [
        column
        for column in df.columns
        if column != id_column
    ]

    if len(feature_columns) != EXPECTED_DACC_FEATURES:

        raise ValueError(
            f"Unexpected number of DACC features: "
            f"{len(feature_columns)}. "
            f"Expected {EXPECTED_DACC_FEATURES}."
        )

    # Convert features to numeric
    dacc_values = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if dacc_values.isnull().any().any():

        raise ValueError(
            "Non-numeric or missing values were detected "
            "in the DACC feature matrix."
        )

    # Check duplicate sequence IDs
    if df[id_column].duplicated().any():

        duplicates = (
            df.loc[
                df[id_column].duplicated(),
                id_column
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate lncRNA IDs found in DACC output: "
            f"{duplicates[:10]}"
        )

    dacc_dict = {}

    for idx, sequence_id in enumerate(
        df[id_column]
    ):

        dacc_dict[str(sequence_id)] = (
            dacc_values
            .iloc[idx]
            .to_numpy(dtype=np.float32)
        )

    return dacc_dict


# ============================================================
# Load DACC + ESM-2 CatBoost model
# ============================================================

def load_dacc_esm_classifier():

    if not os.path.isfile(DACC_MODEL_PATH):

        raise FileNotFoundError(
            f"CatBoost model was not found at:\n"
            f"{DACC_MODEL_PATH}"
        )

    logger.info(
        f"Loading DACC + ESM-2 CatBoost model: "
        f"{DACC_MODEL_PATH}"
    )

    classifier = CatBoostClassifier()

    classifier.load_model(
        DACC_MODEL_PATH
    )

    # Check expected feature count
    model_feature_count = len(
        classifier.get_feature_importance()
    )

    if model_feature_count != (
        EXPECTED_DACC_ESM_FEATURES
    ):

        raise ValueError(
            f"CatBoost model expects "
            f"{model_feature_count} features, "
            f"but the pipeline generates "
            f"{EXPECTED_DACC_ESM_FEATURES}."
        )

    logger.info(
        f"Model expects "
        f"{model_feature_count} features."
    )

    return classifier


# ============================================================
# Load rapid CatBoost model
# ============================================================

def load_rapid_classifier():

    if not os.path.isfile(RAPID_MODEL_PATH):

        raise FileNotFoundError(
            f"Rapid CatBoost model was not found at:\n"
            f"{RAPID_MODEL_PATH}"
        )

    logger.info(
        f"Loading rapid CatBoost model: "
        f"{RAPID_MODEL_PATH}"
    )

    classifier = joblib.load(
        RAPID_MODEL_PATH
    )

    # Verify feature count when supported by the
    # loaded model.
    try:

        model_feature_count = len(
            classifier.feature_names_
        )

    except AttributeError:

        model_feature_count = None

    if (
        model_feature_count is not None
        and model_feature_count
        != EXPECTED_RAPID_FEATURES
    ):

        raise ValueError(
            f"Rapid CatBoost model expects "
            f"{model_feature_count} features, "
            f"but the pipeline generates "
            f"{EXPECTED_RAPID_FEATURES}."
        )

    return classifier


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict lncRNA-protein interactions using "
            "either the rapid composition-based model or "
            "the DACC + ESM-2 t30 model."
        )
    )

    parser.add_argument(
        "-ln",
        "--lncrna_fasta",
        required=True,
        help="Input lncRNA FASTA file."
    )

    parser.add_argument(
        "-pr",
        "--protein_fasta",
        required=True,
        help="Input protein FASTA file."
    )

    parser.add_argument(
        "-o",
        "--output_file",
        default="prediction_results.csv",
        help=(
            "Final prediction output CSV filename. "
            "Default: prediction_results.csv"
        )
    )

    parser.add_argument(
        "-wd",
        "--working_directory",
        default="./",
        help=(
            "Working directory for temporary and final "
            "output files. Default: ./"
        )
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=[
            "rapid",
            "dacc_esm2_t30"
        ],
        default="dacc_esm2_t30",
        help=(
            "Prediction method. "
            "'rapid' uses composition-based features; "
            "'dacc_esm2_t30' uses DACC + ESM-2 t30. "
            "Default: dacc_esm2_t30"
        )
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for interaction "
            "classification. "
            "Probability >= threshold is labeled "
            "'Interacting'; otherwise 'Non-interacting'. "
            "Default: 0.5"
        )
    )

    args = parser.parse_args()


    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    if not 0.0 <= args.threshold <= 1.0:

        raise ValueError(
            "Threshold must be between 0 and 1."
        )


    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not os.path.isfile(
        args.lncrna_fasta
    ):

        raise FileNotFoundError(
            f"LncRNA FASTA file not found:\n"
            f"{args.lncrna_fasta}"
        )

    if not os.path.isfile(
        args.protein_fasta
    ):

        raise FileNotFoundError(
            f"Protein FASTA file not found:\n"
            f"{args.protein_fasta}"
        )


    # --------------------------------------------------------
    # Working directory
    # --------------------------------------------------------

    working_dir = os.path.abspath(
        args.working_directory
    )

    os.makedirs(
        working_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # DACC output
    #
    # Only used for the DACC + ESM-2 method.
    # --------------------------------------------------------

    dacc_output_csv = os.path.join(
        working_dir,
        "dacc_features.csv"
    )


    # --------------------------------------------------------
    # Final prediction output
    # --------------------------------------------------------

    if os.path.isabs(
        args.output_file
    ):

        output_file = args.output_file

    else:

        output_file = os.path.join(
            working_dir,
            args.output_file
        )


    # --------------------------------------------------------
    # Read input FASTA files
    # --------------------------------------------------------

    logger.info(
        "Reading lncRNA FASTA..."
    )

    lncrna_sequences = read_fasta(
        args.lncrna_fasta
    )

    logger.info(
        f"Loaded {len(lncrna_sequences)} "
        f"lncRNA sequences."
    )

    logger.info(
        "Reading protein FASTA..."
    )

    protein_sequences = read_fasta(
        args.protein_fasta
    )

    logger.info(
        f"Loaded {len(protein_sequences)} "
        f"protein sequences."
    )

    if len(lncrna_sequences) == 0:

        raise ValueError(
            "No lncRNA sequences were found."
        )

    if len(protein_sequences) == 0:

        raise ValueError(
            "No protein sequences were found."
        )


    # --------------------------------------------------------
    # Select prediction method
    # --------------------------------------------------------

    logger.info(
        f"Selected prediction method: "
        f"{args.model}"
    )

    logger.info(
        f"Classification threshold: "
        f"{args.threshold}"
    )


    # ========================================================
    # RAPID MODEL
    # ========================================================

    if args.model == "rapid":

        # ----------------------------------------------------
        # Generate rapid features
        # ----------------------------------------------------

        lnc_ids = list(
            lncrna_sequences.keys()
        )

        protein_ids = list(
            protein_sequences.keys()
        )

        lnc_seqs = list(
            lncrna_sequences.values()
        )

        protein_seqs = list(
            protein_sequences.values()
        )

        X = generate_rapid_features(
            lnc_seqs,
            protein_seqs
        )

        total_predictions = (
            len(lnc_ids)
            *
            len(protein_ids)
        )

        logger.info(
            f"Generating {total_predictions:,} "
            "rapid-model predictions..."
        )

        # ----------------------------------------------------
        # Load rapid model
        # ----------------------------------------------------

        classifier = load_rapid_classifier()

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        logger.info(
            "Predicting interaction probabilities..."
        )

        probabilities = classifier.predict_proba(
            X
        )[:, 1]

        # ----------------------------------------------------
        # Build pair IDs
        # ----------------------------------------------------

        lnc_pair_ids = np.repeat(
            lnc_ids,
            len(protein_ids)
        )

        protein_pair_ids = np.tile(
            protein_ids,
            len(lnc_ids)
        )


    # ========================================================
    # DACC + ESM-2 t30 MODEL
    # ========================================================

    else:

        # ----------------------------------------------------
        # Calculate DACC
        # ----------------------------------------------------

        calculate_dacc(
            args.lncrna_fasta,
            dacc_output_csv
        )

        # ----------------------------------------------------
        # Load DACC
        # ----------------------------------------------------

        dacc_features = load_dacc_features(
            dacc_output_csv
        )

        # ----------------------------------------------------
        # Verify DACC coverage
        # ----------------------------------------------------

        missing_lncrnas = [
            sequence_id
            for sequence_id in lncrna_sequences
            if sequence_id not in dacc_features
        ]

        if missing_lncrnas:

            raise ValueError(
                "DACC features were not generated for "
                f"{len(missing_lncrnas)} lncRNA sequences. "
                f"Examples: {missing_lncrnas[:10]}"
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device = get_device()

        logger.info(
            f"Using device: {device}"
        )

        # ----------------------------------------------------
        # Load ESM-2
        # ----------------------------------------------------

        tokenizer, protein_model = (
            load_esm_model(device)
        )

        # ----------------------------------------------------
        # Generate protein embeddings
        # ----------------------------------------------------

        logger.info(
            "Generating ESM-2 protein embeddings..."
        )

        protein_embeddings = {}

        for protein_id, sequence in tqdm(
            protein_sequences.items(),
            desc="Protein embeddings"
        ):

            embedding = (
                generate_protein_embedding(
                    sequence,
                    tokenizer,
                    protein_model,
                    device
                )
            )

            protein_embeddings[
                protein_id
            ] = embedding

        # ----------------------------------------------------
        # Load CatBoost
        # ----------------------------------------------------

        classifier = (
            load_dacc_esm_classifier()
        )

        # ----------------------------------------------------
        # All-by-all prediction
        # ----------------------------------------------------

        total_predictions = (
            len(lncrna_sequences)
            *
            len(protein_sequences)
        )

        logger.info(
            f"Generating {total_predictions:,} "
            "lncRNA-protein predictions..."
        )

        lnc_pair_ids = []
        protein_pair_ids = []
        probabilities = []

        for lncrna_id in tqdm(
            lncrna_sequences,
            desc="Predicting interactions"
        ):

            lnc_dacc = dacc_features[
                lncrna_id
            ]

            for protein_id in protein_sequences:

                protein_embedding = (
                    protein_embeddings[
                        protein_id
                    ]
                )

                # --------------------------------------------
                # 288 DACC + 640 ESM-2 = 928
                # --------------------------------------------

                feature_vector = np.concatenate(
                    [
                        lnc_dacc,
                        protein_embedding
                    ]
                ).astype(
                    np.float32
                )

                if feature_vector.shape[0] != (
                    EXPECTED_DACC_ESM_FEATURES
                ):

                    raise ValueError(
                        f"Unexpected feature vector size: "
                        f"{feature_vector.shape[0]}. "
                        f"Expected "
                        f"{EXPECTED_DACC_ESM_FEATURES}."
                    )

                feature_vector = (
                    feature_vector.reshape(1, -1)
                )

                probability = (
                    classifier
                    .predict_proba(
                        feature_vector
                    )[0][1]
                )

                lnc_pair_ids.append(
                    lncrna_id
                )

                protein_pair_ids.append(
                    protein_id
                )

                probabilities.append(
                    float(probability)
                )


    # ========================================================
    # Classification
    # ========================================================

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    predicted_labels = np.where(
        probabilities >= args.threshold,
        "Interacting",
        "Non-interacting"
    )


    # ========================================================
    # Save results
    # ========================================================

    results_df = pd.DataFrame(
        {
            "LncRNA_ID": lnc_pair_ids,
            "Protein_ID": protein_pair_ids,
            "Interaction_Probability": np.round(
                probabilities,
                4
            ),
            "Predicted_Label": predicted_labels
        }
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Prediction results written to:\n"
        f"{output_file}"
    )

    logger.info(
        "Prediction completed successfully."
    )

    logger.info(
        f"Interacting predictions: "
        f"{np.sum(predicted_labels == 'Interacting'):,}"
    )

    logger.info(
        f"Non-interacting predictions: "
        f"{np.sum(predicted_labels == 'Non-interacting'):,}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
train_final.py
==============
Train a final CatBoost model using the manually selected feature set and
hyperparameters on the combined training + validation dataset.

The training and validation datasets are merged and used together for
final model fitting. No cross-validation, seed averaging, or hyperparameter
search is performed.

Usage:
------
    python train_final.py

Outputs:
--------
    ./results/catboost_final_model_full.cbm
        -> final CatBoost model trained on training + validation data

    ./results/final_training_info.json
        -> feature set, hyperparameters, and dataset information
"""

import json
import os

import pandas as pd
from catboost import CatBoostClassifier


# --- 1. CONFIGURATION ----------------------------------------------------

TRAINING_SET_PATH = "./dataset/lncrnapi_train_labeled.tsv"
VALIDATION_SET_PATH = "./dataset/lncrnapi_val_labeled.tsv"

LNCRNA_FEATURE_DIR = "./features/lncRNA/"
TARGET_FEATURE_DIR = "./features/targets/"

# --- Manually chosen feature set ---
LNCRNA_FEATURE = "DACC"
TARGET_FEATURE = "esm2.t30.150M.UR50D"

# --- Manually chosen hyperparameters ---
HYPERPARAMETERS = {'iterations': 1032, 'learning_rate': 0.04263494650162085, 'depth': 8, 'l2_leaf_reg': 2.37209479411584, 'bagging_temperature': 0.5696056806946824, 'random_strength': 0.6814225207851792, 'border_count': 173, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 17}

RESULTS_DIR = "./results"

MODEL_FILE = os.path.join(
    RESULTS_DIR,
    "catboost_final_model_full.cbm"
)

INFO_FILE = os.path.join(
    RESULTS_DIR,
    "final_training_info.json"
)


# --- 2. DATA LOADING / MERGING --------------------------------------------

def load_feature_file(
    directory,
    prefix_template,
    feature_name,
    id_col,
    rename_from="Sequence_ID"
):
    """
    Load one feature CSV and standardize its ID column.
    """

    file_path = os.path.join(
        directory,
        prefix_template.format(feature_name)
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    df = pd.read_csv(file_path).rename(
        columns={rename_from: id_col}
    )

    if pd.api.types.is_string_dtype(df[id_col]):
        df[id_col] = df[id_col].str.lstrip(">")

    return df


def create_combined_dataset(
    interaction_df,
    lncrna_features_df,
    target_features_df
):
    """
    Merge interaction pairs with lncRNA and target features.
    """

    missing_nc = (
        set(interaction_df["ncID"])
        - set(lncrna_features_df["ncID"])
    )

    if missing_nc:
        print(
            f"\n⚠️ Missing lncRNA features for "
            f"{len(missing_nc)} IDs."
        )

    missing_tar = (
        set(interaction_df["tarID"])
        - set(target_features_df["tarID"])
    )

    if missing_tar:
        print(
            f"\n⚠️ Missing target features for "
            f"{len(missing_tar)} IDs."
        )

    # Merge lncRNA features
    merged_df = pd.merge(
        interaction_df,
        lncrna_features_df,
        on="ncID",
        how="left"
    )

    # Merge target features
    final_df = pd.merge(
        merged_df,
        target_features_df,
        on="tarID",
        how="left"
    )

    # Check for missing values
    if final_df.isnull().values.any():
        print(
            "\n⚠️ Null values exist after merging; "
            "filling with 0."
        )
        final_df.fillna(0, inplace=True)

    # Move label to the end
    label_column = final_df.pop("label")

    # Remove sequence IDs
    final_df = final_df.drop(
        columns=["ncID", "tarID"]
    )

    final_df["label"] = label_column

    return final_df


# --- 3. MAIN ---------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------------------------------------------------
    # Load training and validation interaction datasets
    # ----------------------------------------------------------------------

    try:

        train_interactions = pd.read_csv(
            TRAINING_SET_PATH,
            sep="\t"
        )

        val_interactions = pd.read_csv(
            VALIDATION_SET_PATH,
            sep="\t"
        )

        print(
            f"✅ Loaded interactions:"
            f"\n   Training:   {len(train_interactions):,} rows"
            f"\n   Validation: {len(val_interactions):,} rows"
        )

    except FileNotFoundError as e:

        raise SystemExit(
            f"❌ FATAL ERROR: Could not load interaction file: {e}"
        )


    # ----------------------------------------------------------------------
    # Load selected feature files
    # ----------------------------------------------------------------------

    print("\n📂 Loading feature files...")

    lnc_df = load_feature_file(
        LNCRNA_FEATURE_DIR,
        "total_{}_lncrna.csv",
        LNCRNA_FEATURE,
        id_col="ncID"
    )

    tar_df = load_feature_file(
        TARGET_FEATURE_DIR,
        "total_comp_{}_targets.csv",
        TARGET_FEATURE,
        id_col="tarID"
    )

    print(f"   lncRNA feature: {LNCRNA_FEATURE}")
    print(f"   Target feature: {TARGET_FEATURE}")


    # ----------------------------------------------------------------------
    # Create feature datasets separately
    # ----------------------------------------------------------------------

    print("\n🔗 Creating training dataset...")

    train_df = create_combined_dataset(
        train_interactions,
        lnc_df,
        tar_df
    )

    print("\n🔗 Creating validation dataset...")

    val_df = create_combined_dataset(
        val_interactions,
        lnc_df,
        tar_df
    )


    # ----------------------------------------------------------------------
    # Combine training + validation datasets
    # ----------------------------------------------------------------------

    print("\n🔄 Combining training + validation datasets...")

    full_df = pd.concat(
        [train_df, val_df],
        axis=0,
        ignore_index=True
    )

    print(
        f"   Training samples:   {len(train_df):,}"
        f"\n   Validation samples: {len(val_df):,}"
        f"\n   Final dataset:      {len(full_df):,}"
    )


    # ----------------------------------------------------------------------
    # Prepare X and y
    # ----------------------------------------------------------------------

    X_full = full_df.drop(
        "label",
        axis=1
    )

    y_full = full_df["label"]


    # ----------------------------------------------------------------------
    # Display class distribution
    # ----------------------------------------------------------------------

    print("\n📊 Final dataset class distribution:")

    class_counts = y_full.value_counts()

    for label, count in class_counts.items():

        percentage = (
            count / len(y_full)
        ) * 100

        print(
            f"   Class {label}: "
            f"{count:,} ({percentage:.2f}%)"
        )


    # ----------------------------------------------------------------------
    # Feature-set name
    # ----------------------------------------------------------------------

    feature_set_name = (
        f"{LNCRNA_FEATURE}-{TARGET_FEATURE}"
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        "🚀 TRAINING FINAL CATBOOST MODEL"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"Feature set : {feature_set_name}"
        f"\nSamples     : {len(X_full):,}"
        f"\nFeatures    : {X_full.shape[1]}"
    )

    print(
        "\nThe model will be trained on "
        "TRAINING + VALIDATION data."
    )


    # ----------------------------------------------------------------------
    # Train final model
    # ----------------------------------------------------------------------

    model = CatBoostClassifier(
        **HYPERPARAMETERS,
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_full,
        y_full
    )


    # ----------------------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------------------

    model.save_model(
        MODEL_FILE
    )


    # ----------------------------------------------------------------------
    # Save training information
    # ----------------------------------------------------------------------

    training_info = {

        "feature_set": feature_set_name,

        "lncrna_feature": LNCRNA_FEATURE,

        "target_feature": TARGET_FEATURE,

        "training_samples": int(len(train_df)),

        "validation_samples": int(len(val_df)),

        "final_training_samples": int(len(full_df)),

        "number_of_features": int(X_full.shape[1]),

        "class_distribution": {
            str(k): int(v)
            for k, v in class_counts.items()
        },

        "hyperparameters": HYPERPARAMETERS,

        "random_seed": 42,

        "training_data": (
            "Training + Validation"
        ),

        "model_type": "CatBoostClassifier"
    }


    with open(
        INFO_FILE,
        "w"
    ) as f:

        json.dump(
            training_info,
            f,
            indent=2
        )


    # ----------------------------------------------------------------------
    # Final messages
    # ----------------------------------------------------------------------

    print(
        f"\n{'=' * 70}"
    )

    print(
        "✅ FINAL MODEL TRAINING COMPLETE"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"\n📁 Model saved to:"
        f"\n   {MODEL_FILE}"
    )

    print(
        f"\n📁 Training information saved to:"
        f"\n   {INFO_FILE}"
    )

    print(
        "\n⚠️ No validation metrics were calculated because "
        "the validation set was included in final training."
    )

    print(
        "\nThe saved model is now ready for evaluation "
        "on an independent test set or for deployment."
    )


# In[ ]:





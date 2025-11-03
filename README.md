# 🧬 lncrna-PI - LncRNA–Protein Interaction Prediction

lncrnaPI is a command-line tool for predicting **lncRNA–Protein interactions** using **pre-trained language models (DNABERT-2 and ESM-2)** for sequence embedding and a **CatBoost classifier** for interaction probability estimation.

It supports two modes:
- **Rapid** — based on sequence composition features (fast, lightweight)
- **LLM** — based on transformer embeddings (DNABERT2 + ESM2)

The script performs **all-by-all predictions** between every lncRNA and every protein sequence in the provided FASTA files.

---

## 📦 Features

- Vectorized and efficient FASTA parsing  
- All-by-all pairing of lncRNA and protein sequences  
- Automatic **feature extraction**:
  - **Rapid mode:** nucleotide and amino acid composition (%)
  - **LLM mode:** transformer-based embeddings (DNABERT2 + ESM2)
- Automatic model selection:
  - `catboost_model_rapid.joblib` → Composition model
  - `catboost_dnabert2_esm-t30.joblib` → Embedding model
- **GPU-aware** embedding generation (with safe fallback to CPU)
- Generates probability and binary interaction predictions

---

## 🧰 Dependencies

Install the following dependencies before running the script:

```bash
pip install torch==2.6.0 transformers==4.57.0 catboost==1.2.8 joblib tqdm numpy pandas
```
---

## ⚙️ Usage

### 1️⃣ **Rapid (Composition-Based) Prediction**

This mode uses simple % composition features (very fast).

```bash
python lncrnapi_cli.py   -lf ./data/example_lncRNA.fasta   -pf ./data/example_protein.fasta   -wd ./output   -model rapid
```

**Model used:**  
`./model/catboost_model_rapid.joblib`

---

### 2️⃣ **LLM (Embedding-Based) Prediction**

This mode uses transformer embeddings from **DNABERT2** (for lncRNA) and **ESM2-T30** (for protein).

```bash
python lncrnapi_cli.py  -lf ./data/example_lncRNA.fasta   -pf ./data/example_protein.fasta   -wd ./output   -model llm
```

**Model used:**  
`./model/catboost_dnabert2_esm-t30.joblib`

---

### **Arguments**

| Argument | Description | Required |
|-----------|--------------|-----------|
| `-lf` | Path to the FASTA file containing lncRNA sequences. | ✅ |
| `-pf` | Path to the FASTA file containing protein sequences. | ✅ |
| `-wd` | Path to the working directory. | ✅ |
| `-model` | Choice of model to be used. | ✅ |
| `-t` | Threshold | ❌ |
---

## 💾 Output

A CSV file named `output.csv` is generated in the output directory:

| lncRNA_ID | Protein_ID | Interaction_Probability | Predicted_Label |
|------------|-------------|--------------------------|------------------|
| lnc1 | P12345 | 0.87 | 1 |
| lnc1 | P67890 | 0.34 | 0 |
| ... | ... | ... | ... |

- **Interaction_Probability:** Probability predicted by CatBoost  
- **Predicted_Label:** 1 → interaction, 0 → non-interaction

---

## ⚡ Hardware Acceleration

The script automatically detects and uses available hardware:

- ✅ **CUDA GPU** (NVIDIA)
- ✅ **MPS** (Apple Silicon)
- ⚠️ **CPU** (fallback)

---

## 📜 Citation

If you use this tool in your research, please cite:

> **Your Name et al.**  
> *A Deep Learning Framework for lncRNA–Protein Interaction Prediction Using Transformer-Based Sequence Embeddings* (2025)

---

## 🧩 Repository Structure

```
├── lncrnapi_cli.py       # Main CLI script
├── README.md
├── LICENSE
├── test_lncrna.fa
├── test_protein.fa
├── output.csv 
└── models/
    ├── catboost_model_rapid.joblib
    └── catboost_dnabert2_esm-t30.joblib                
```

---

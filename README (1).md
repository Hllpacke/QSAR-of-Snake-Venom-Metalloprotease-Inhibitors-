# SVMP IC50 Classification Models

This folder contains cleaned, GitHub-ready Python scripts extracted and generalised from four SVMP classification notebooks:

- `BJAR_ClassificationModel_FinalScaled.ipynb`
- `CA_ClassificationModel_FinalScaled.ipynb`
- `CR_ClassificationModel_FinalScaled.ipynb`
- `EOC_ClassificationModel_FinalScaled.ipynb`

The original notebooks used hard-coded local paths and dataset filenames. Those have been removed. The scripts now accept input files through command-line arguments and save reproducible outputs to a chosen results folder.

## What the scripts do

The cleaned workflow follows the notebook logic:

1. Load a CSV, or a CSV stored inside a zip archive.
2. Standardise the SMILES and binary activity columns.
3. Compute all RDKit descriptors plus Morgan fingerprint density.
4. Plot five basic descriptor distributions:
   - molecular weight
   - MolLogP
   - H-bond donors
   - H-bond acceptors
   - Morgan fingerprint density
5. Apply an 80:20 stratified train/test split.
6. Apply SMOTE to the training set.
7. Scale descriptors using `StandardScaler`.
8. Remove highly correlated descriptors.
9. Train four classifiers:
   - Random Forest
   - Logistic Regression
   - SVM
   - KNN
10. Evaluate accuracy, ROC AUC and MCC.
11. Run y-randomisation for AUC and MCC.
12. Save confusion matrices, ROC plots, random-vs-actual bar plots, tables and trained models.
13. Optionally run Kernel SHAP for Logistic Regression, SVM and KNN.

## Files

| File | Purpose |
|---|---|
| `svmp_classification_utils.py` | Shared descriptor, modelling, plotting and y-randomisation functions |
| `train_svmp_classifier.py` | Generic training script for BJAR, CA, CR or EOC |
| `train_bjar_classifier.py` | BJAR convenience wrapper |
| `train_ca_classifier.py` | CA convenience wrapper |
| `train_cr_classifier.py` | CR convenience wrapper |
| `train_eoc_classifier.py` | EOC convenience wrapper |
| `run_all_svmp_models.py` | Optional batch runner from a manifest CSV |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps data/results/models out of Git by default |

## Installation

A conda environment is recommended because RDKit is easiest to install from conda-forge:

```bash
conda create -n svmp-classification python=3.10 rdkit -c conda-forge
conda activate svmp-classification
pip install -r requirements.txt
```

## Example: run one model from a CSV

```bash
python train_bjar_classifier.py \
  --data-csv data/bjar_training.csv \
  --smiles-col SMILES \
  --activity-col BJAR_IC50_2Activity \
  --output-dir results/bjar
```

Equivalent generic call:

```bash
python train_svmp_classifier.py \
  --task BJAR \
  --data-csv data/bjar_training.csv \
  --smiles-col SMILES \
  --activity-col BJAR_IC50_2Activity \
  --output-dir results/bjar
```

## Example: run from a zip archive

```bash
python train_svmp_classifier.py \
  --task CA \
  --data-zip data/svmp_ic50_data.zip \
  --zip-member CA.csv \
  --smiles-col Smiles \
  --activity-col CA_IC50_Activity \
  --output-dir results/ca
```

## Example: run all four models

Create `data/model_manifest.csv`:

```csv
task,data_csv,activity_col,smiles_col
BJAR,data/bjar_training.csv,BJAR_IC50_2Activity,SMILES
CA,data/ca_training.csv,CA_IC50_Activity,SMILES
CR,data/cr_training.csv,CR_IC50_Activity,SMILES
EOC,data/eoc_training.csv,EOC_IC50_2Activity,SMILES
```

Then run:

```bash
python run_all_svmp_models.py \
  --manifest data/model_manifest.csv \
  --output-dir results
```

## Optional SHAP analysis

Kernel SHAP can be slow, especially for SVM and KNN. It is disabled by default.

```bash
python train_svmp_classifier.py \
  --task EOC \
  --data-csv data/eoc_training.csv \
  --activity-col EOC_IC50_2Activity \
  --output-dir results/eoc \
  --run-shap
```

## Expected input format

At minimum, your input CSV should contain:

```csv
SMILES,Activity
CCO,0
CCN,1
```

Or use the original activity column names, for example:

```csv
Smiles,BJAR_IC50_2Activity
CCO,0
CCN,1
```

## Notes for public GitHub release

- No private data are included.
- No local paths are included.
- `data/`, `results/`, `models/` and common binary model artefacts are ignored by Git.
- The scripts are syntax-checked but require your real input data to run end-to-end.

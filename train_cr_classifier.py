"""Convenience wrapper for the CR SVMP classification model.

Supply your own data path at runtime. No local/private notebook paths are used.

Example:
    python train_cr_classifier.py --data-csv data/cr_training.csv --output-dir results/cr
"""

from train_svmp_classifier import main


if __name__ == "__main__":
    main(default_task="CR", default_activity_col="CR_IC50_Activity")

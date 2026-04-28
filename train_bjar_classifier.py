"""Convenience wrapper for the BJAR SVMP classification model.

Supply your own data path at runtime. No local/private notebook paths are used.

Example:
    python train_bjar_classifier.py --data-csv data/bjar_training.csv --output-dir results/bjar
"""

from train_svmp_classifier import main


if __name__ == "__main__":
    main(default_task="BJAR", default_activity_col="BJAR_IC50_2Activity")

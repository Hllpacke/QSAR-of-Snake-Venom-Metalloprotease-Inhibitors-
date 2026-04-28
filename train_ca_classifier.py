"""Convenience wrapper for the CA SVMP classification model.

Supply your own data path at runtime. No local/private notebook paths are used.

Example:
    python train_ca_classifier.py --data-csv data/ca_training.csv --output-dir results/ca
"""

from train_svmp_classifier import main


if __name__ == "__main__":
    main(default_task="CA", default_activity_col="CA_IC50_Activity")

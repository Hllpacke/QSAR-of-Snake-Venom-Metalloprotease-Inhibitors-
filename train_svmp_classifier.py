"""
Train a scaled SVMP IC50 activity classifier.

This script is a cleaned, GitHub-ready version of the SVMP classification
notebooks for BJAR, CA, CR and EOC data. It removes hard-coded local paths and
requires data to be supplied as a CSV or as a CSV inside a zip archive.

Example:
    python train_svmp_classifier.py \
        --task BJAR \
        --data-csv data/bjar_training.csv \
        --smiles-col SMILES \
        --activity-col BJAR_IC50_2Activity \
        --output-dir results/bjar
"""

from __future__ import annotations

import argparse
from pathlib import Path

from svmp_classification_utils import (
    build_descriptor_frame,
    compute_shap_importance_optional,
    DEFAULT_TASK_LABEL_COLUMNS,
    ensure_output_dir,
    normalise_input_columns,
    plot_basic_descriptor_histograms,
    print_summary,
    random_forest_feature_importance,
    read_training_table,
    run_classification_workflow,
    save_workflow_artifacts,
)


def build_parser(
    default_task: str | None = None,
    default_activity_col: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SVMP activity classifiers from SMILES and binary activity labels."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data-csv", type=str, help="Path to a CSV file containing SMILES and activity labels.")
    input_group.add_argument("--data-zip", type=str, help="Path to a zip archive containing a CSV file.")

    parser.add_argument(
        "--zip-member",
        type=str,
        default=None,
        help="CSV filename inside --data-zip, e.g. BJAR.csv. Required if --data-zip is used.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=default_task,
        choices=["BJAR", "CA", "CR", "EOC"],
        help="Optional task/model family. Used only for naming outputs and default label-column inference.",
    )
    parser.add_argument("--smiles-col", type=str, default="SMILES", help="Name of the SMILES column in the input data.")
    parser.add_argument(
        "--activity-col",
        type=str,
        default=default_activity_col,
        help=(
            "Name of the binary activity label column. If omitted, the script tries a task-specific "
            "default and then common alternatives such as Activity/Label."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results/svmp_classifier", help="Directory for outputs.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output files. Defaults to --task or 'svmp'.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--corr-thresh", type=float, default=0.90, help="Absolute correlation threshold for feature removal.")
    parser.add_argument("--n-randomisations", type=int, default=50, help="Number of y-randomisation permutations.")
    parser.add_argument("--fingerprint-radius", type=int, default=2, help="Morgan fingerprint radius for density feature.")
    parser.add_argument("--fingerprint-bits", type=int, default=1024, help="Morgan fingerprint bit length.")
    parser.add_argument("--run-shap", action="store_true", help="Run optional Kernel SHAP analysis. This can be slow.")
    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=50,
        help="Number of training samples used as Kernel SHAP background.",
    )

    return parser


def main(
    default_task: str | None = None,
    default_activity_col: str | None = None,
) -> None:
    parser = build_parser(default_task=default_task, default_activity_col=default_activity_col)
    args = parser.parse_args()

    task = args.task.upper() if args.task else None
    prefix = args.prefix or (task.lower() if task else "svmp")
    output_dir = ensure_output_dir(args.output_dir)

    # Use a task-specific default label column if available and the user did not provide one.
    activity_col = args.activity_col
    if activity_col is None and task:
        activity_col = DEFAULT_TASK_LABEL_COLUMNS.get(task)

    raw_df = read_training_table(
        data_csv=args.data_csv,
        data_zip=args.data_zip,
        zip_member=args.zip_member,
    )
    clean_df = normalise_input_columns(
        raw_df,
        smiles_col=args.smiles_col,
        activity_col=activity_col,
        task=task,
    )

    descriptor_df = build_descriptor_frame(
        clean_df,
        radius=args.fingerprint_radius,
        n_bits=args.fingerprint_bits,
    )
    descriptor_df.to_csv(output_dir / "tables" / f"{prefix}_rdkit_descriptors.csv", index=False)

    plot_basic_descriptor_histograms(descriptor_df, output_dir=output_dir, prefix=prefix)

    result = run_classification_workflow(
        descriptor_df,
        output_dir=output_dir,
        prefix=prefix,
        test_size=args.test_size,
        random_state=args.random_state,
        corr_thresh=args.corr_thresh,
        n_randomisations=args.n_randomisations,
    )

    random_forest_feature_importance(
        result.trained_models,
        result.feature_names,
        output_dir=output_dir,
        prefix=prefix,
    )

    if args.run_shap:
        compute_shap_importance_optional(
            result.trained_models,
            result.X_train_final,
            result.X_test_final,
            result.feature_names,
            output_dir=output_dir,
            prefix=prefix,
            background_size=args.shap_background_size,
        )

    save_workflow_artifacts(
        result,
        descriptor_df,
        output_dir=output_dir,
        prefix=prefix,
        descriptor_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        corr_thresh=args.corr_thresh,
    )

    print_summary(output_dir=output_dir, prefix=prefix, final_table=result.final_table)


if __name__ == "__main__":
    main()

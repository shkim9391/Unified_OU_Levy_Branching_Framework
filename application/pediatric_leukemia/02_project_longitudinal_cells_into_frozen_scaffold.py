from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oulb.projection import (  # noqa: E402
    CoarseStateDefinition,
    attach_sample_scores_to_anndata,
    compute_coarse_state_scores,
    merge_sample_score_tables,
    project_samples_from_manifest,
    require_columns,
)
from oulb.scaffold import load_frozen_scaffold_from_anndata  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "pediatric_leukemia_stage1.toml"

QUERY_REQUIRED_COLUMNS = (
    "sample_id",
    "patient_id",
    "clinical_timepoint_raw",
    "clinical_timepoint_coarse",
    "Classified_Celltype",
)

STATE_DEFINITION = CoarseStateDefinition(
    main_states={
        "state_HSC": ("HSC",),
        "state_Prog": ("Progenitor",),
        "state_GMP": ("GMP",),
        "state_MonoDC": ("Monocytes", "cDC"),
    },
    auxiliary_states={
        "aux_EryBaso": ("Early.Basophil", "Early.Erythrocyte"),
        "aux_CLP": ("CLP",),
    },
    branch_labels={
        "state_HSC": "HSC-like basin",
        "state_Prog": "Progenitor-like basin",
        "state_GMP": "GMP-like basin",
        "state_MonoDC": "Mono/DC-like basin",
    },
)

PROJECTION_NOTE = (
    "PC1/PC2 reconstructed from frozen normal-cell broad-group PCA; "
    "malignant coarse-state summaries reconstructed from malignant "
    "Classified_Celltype fractions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Stage 1 TOML configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Stage 1 output files.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def require_input(path: Path) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_output(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\nRe-run with --overwrite to replace it."
        )
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["projection"]

    reference_h5ad = require_input(Path(paths["reference_h5ad"]))
    query_h5ad = require_input(Path(paths["query_h5ad"]))
    manifest_csv = require_input(Path(paths["manifest_csv"]))
    metadata_root = require_input(Path(paths["metadata_root"]))
    output_h5ad = prepare_output(
        Path(paths["output_projected_h5ad"]), overwrite=args.overwrite
    )
    output_sample_csv = prepare_output(
        Path(paths["output_sample_scores_csv"]), overwrite=args.overwrite
    )

    reference_scores_value = str(paths.get("reference_scores_csv", "")).strip()
    reference_scores_path = (
        require_input(Path(reference_scores_value)) if reference_scores_value else None
    )

    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - application environment check
        raise SystemExit(
            "This wrapper requires anndata. Install the application dependencies with:\n"
            "  python -m pip install -e '.[application]'"
        ) from exc

    reference = ad.read_h5ad(reference_h5ad)
    query = ad.read_h5ad(query_h5ad)
    manifest = pd.read_csv(manifest_csv)

    require_columns(query.obs, QUERY_REQUIRED_COLUMNS, context="Query AnnData.obs")
    scaffold = load_frozen_scaffold_from_anndata(
        reference,
        reference_scores_path=reference_scores_path,
        coordinate_columns=("PC1", "PC2"),
        label_column="ecotype_label",
        reference_id_column="sample_id",
    )

    query_sample_ids = set(query.obs["sample_id"].astype(str).unique())
    context_scores = project_samples_from_manifest(
        manifest,
        query_sample_ids,
        scaffold,
        metadata_root=metadata_root,
        metadata_read_kwargs={
            "sep": str(paths.get("metadata_separator", "\t")),
            "compression": str(paths.get("metadata_compression", "gzip")),
        },
        sample_column="sample_id",
        metadata_file_column="metadata_file",
        manifest_passthrough={
            "patient_id": "patient_id_manifest",
            "clinical_timepoint_raw": "clinical_timepoint_raw_manifest",
            "clinical_timepoint_coarse": "clinical_timepoint_coarse_manifest",
        },
        malignancy_column="Malignant",
        cell_type_column="Classified_Celltype",
        excluded_malignancy_values=("Malignant",),
        fraction_prefix="normal_frac__",
    )

    state_scores = compute_coarse_state_scores(
        query.obs,
        STATE_DEFINITION,
        sample_column="sample_id",
        cell_type_column="Classified_Celltype",
        count_column="n_malignant_cells",
        branch_id_column="branch_id",
        branch_max_probability_column="branch_maxprob",
        branch_entropy_column="branch_entropy",
    )
    sample_scores = merge_sample_score_tables(
        context_scores,
        state_scores,
        sample_column="sample_id",
        expected_sample_ids=query_sample_ids,
    )

    # Preserve the validated behavior: write the sample table before attaching
    # the values to every malignant cell.
    sample_scores.to_csv(output_sample_csv, index=False)

    required_scores = [
        "PC1",
        "PC2",
        *STATE_DEFINITION.scaffold_columns,
        "ecotype_label",
        "branch_id",
        "branch_maxprob",
        "branch_entropy",
    ]
    query = attach_sample_scores_to_anndata(
        query,
        sample_scores,
        sample_column="sample_id",
        required_score_columns=required_scores,
        coordinate_columns=("PC1", "PC2"),
        scaffold_columns=STATE_DEFINITION.scaffold_columns,
        coordinate_obsm_key="X_fig2",
        scaffold_obsm_key="X_scaffold",
        projected_flag_column="sample_has_projected_scores",
        note_key="figure3_projection_note",
        note=PROJECTION_NOTE,
        scaffold_feature_order_key="figure3_scaffold_feature_order",
        overwrite_existing=False,
    )
    query.write_h5ad(output_h5ad)

    print(f"[DONE] Saved projected query object: {output_h5ad}")
    print(f"[DONE] Saved sample-level score table: {output_sample_csv}")
    print("\n[SUMMARY] samples with projected scores by timepoint")
    print(
        query.obs.groupby("clinical_timepoint_coarse", observed=True)["sample_id"]
        .nunique()
        .sort_index()
    )


if __name__ == "__main__":
    main()
